"""
metro_st_v9_seasonal_baseline_10min.py
10分钟粒度 · 季节编码基线 + 时空残差修正 + 外部特征

范式: ES-RNN / N-BEATS 风格
  基线模型: SeasonalEncoder(7天滞后) — 一个小型 MLP，从历史同期学习周期模式
  残差模型: ResGCN + TemporalTransformer — 捕捉基线无法解释的时空变异
  外部模型: MLP 小分支 — 天气/空气质量作为环境调制

输出 = 可学习季节基线 + 时空残差
"""
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.layers import (
    Input, Dense, Layer, MultiHeadAttention, LayerNormalization,
    Dropout, Reshape, RepeatVector, Concatenate, Lambda,
    BatchNormalization, Add
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================================================
# 1. 数据加载 + 10min聚合 + 外部特征
# ======================================================

df = pd.read_csv(r"D:\PythonProject4\metro_flow_timeseries_ready_v22.csv")
df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":" +
    df["minute"].astype(str).str.zfill(2)
)
df["dt10"] = df["datetime"].dt.floor("10min")
df = df.groupby(["dt10", "stationCode"], as_index=False).agg({
    "entry_interval": "sum", "exit_interval": "sum",
    "stationName": "first", "lineName": "first",
    "date": lambda x: x.iloc[0],
    "hour": lambda x: x.iloc[0],
    "minute": lambda x: (x.iloc[0] // 10) * 10,
})
df.rename(columns={"dt10": "datetime"}, inplace=True)

df_ext = pd.read_csv(r"D:\PythonProject4\chengdu_external_factors.csv")
df_ext["datetime"] = pd.to_datetime(df_ext["datetime"])
df_ext.drop(columns=["visibility_m"], inplace=True, errors="ignore")
df_ext["wind_dir_sin"] = np.sin(2 * np.pi * df_ext["wind_direction_deg"] / 360.0)
df_ext["wind_dir_cos"] = np.cos(2 * np.pi * df_ext["wind_direction_deg"] / 360.0)
ext_cols = [c for c in df_ext.columns if c not in ["datetime", "wind_direction_deg"]]
df_ext["dt10"] = df_ext["datetime"].dt.floor("10min")
df_ext10 = df_ext.groupby("dt10")[ext_cols].mean().reset_index()
df_ext10.rename(columns={"dt10": "datetime"}, inplace=True)
df = df.merge(df_ext10, on="datetime", how="left")
print(f"合并后 shape: {df.shape}")

# ======================================================
# 2. 时间特征 + 站点
# ======================================================

STEPS_PER_DAY = 144  # 24h * 6 (10min interval)
df["weekday"] = df["datetime"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
df["time_index"] = df["datetime"].dt.hour * 6 + df["datetime"].dt.minute // 10
df["sin_time"] = np.sin(2 * np.pi * df["time_index"] / STEPS_PER_DAY)
df["cos_time"] = np.cos(2 * np.pi * df["time_index"] / STEPS_PER_DAY)
df["is_peak"] = (
    ((df["time_index"] >= 42) & (df["time_index"] <= 54)) |
    ((df["time_index"] >= 102) & (df["time_index"] <= 114))
).astype(int)
holiday_list = ["2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03"]
df["is_holiday"] = (df["datetime"].dt.date.astype(str).isin(holiday_list)).astype(int)
df["date_str"] = df["datetime"].dt.date.astype(str)
tomorrow_map = {}
for i, d in enumerate(sorted(df["date_str"].unique())):
    tomorrow_map[d] = 1 if (i + 1 < len(sorted(df["date_str"].unique())) and sorted(df["date_str"].unique())[i + 1] in holiday_list) else 0
df["is_tomorrow_holiday"] = df["date_str"].map(tomorrow_map)

dead_stations = {835, 836, 930, 1824}
station_order = [
    130, 132, 133, 134, 135, 136, 137, 138, 139,
    546, 547, 548, 549, 550, 551, 552, 553, 554,
    625, 626, 628, 655, 656, 657, 658, 659, 660, 661, 662,
    736, 739, 833, 834, 838, 839, 840, 1823
]
station_order = [s for s in station_order if s not in dead_stations]
NODES = len(station_order)
print(f"有效站点数: {NODES}")
df = df[df["stationCode"].isin(station_order)].copy()

A = np.zeros((NODES, NODES))
lines = [
    [130, 132, 133, 134, 135, 136, 137, 138, 139],
    [546, 547, 548, 549, 550, 551, 552, 553, 554],
    [625, 626, 628, 655, 656, 657, 658, 659, 660, 661, 662],
    [736, 739], [833, 834, 838, 839, 840], [1823]
]
station_to_idx = {s: i for i, s in enumerate(station_order)}
for line in lines:
    for i in range(len(line) - 1):
        a, b = station_to_idx[line[i]], station_to_idx[line[i + 1]]
        A[a, b] = 1; A[b, a] = 1
for s1, s2 in [(136, 1823)]:
    if s1 in station_to_idx and s2 in station_to_idx:
        i, j = station_to_idx[s1], station_to_idx[s2]
        A[i, j] = 1; A[j, i] = 1
A_self = A + np.eye(NODES)
D = np.diag(np.power(A_self.sum(axis=1), -0.5))
A_hat = D @ A_self @ D

# ======================================================
# 3. 特征列表
# ======================================================

flow_features = ["entry_interval", "exit_interval"]
time_features = ["sin_time", "cos_time", "weekday", "is_weekend", "is_peak",
                 "is_holiday", "is_tomorrow_holiday"]
external_features = [
    "temp_c", "humidity_pct", "precip_mm", "rain_mm",
    "weather_code", "wind_speed_kmh", "wind_dir_sin", "wind_dir_cos",
    "pressure_hpa", "pm25", "pm10", "co_ugm3", "no2_ugm3", "so2_ugm3",
    "o3_ugm3", "european_aqi", "us_aqi", "dust_ugm3"
]

TIME_STEPS = 12  # 2小时 (12 × 10min)
SEASONAL_LAGS_DAYS = [1, 2, 7]
SEASONAL_LAGS = [d * STEPS_PER_DAY for d in SEASONAL_LAGS_DAYS]  # [144, 288, 1008]
SEASONAL_DIM = 6  # EMA基线(2) + 昨天(2) + 上周(2)

# ======================================================
# 4. 标准化
# ======================================================

unique_datetimes = sorted(df["datetime"].unique())
n_total = len(unique_datetimes)
train_end_idx = int(n_total * 0.7)
val_end_idx   = int(n_total * 0.85)
train_dts = unique_datetimes[:train_end_idx]
val_dts   = unique_datetimes[train_end_idx:val_end_idx]
test_dts  = unique_datetimes[val_end_idx:]
print(f"时间点: {n_total}  训练: {len(train_dts)}  验证: {len(val_dts)}  测试: {len(test_dts)}")

scaler_flow = MinMaxScaler(); scaler_time = MinMaxScaler(); scaler_ext = MinMaxScaler()
train_boundary = train_dts[-1]; train_mask = df["datetime"] <= train_boundary
scaler_flow.fit(df.loc[train_mask, flow_features])
scaler_time.fit(df.loc[train_mask, time_features])
scaler_ext.fit(df.loc[train_mask, external_features])
df[flow_features] = scaler_flow.transform(df[flow_features])
df[time_features] = scaler_time.transform(df[time_features])
df[external_features] = scaler_ext.transform(df[external_features])

# ======================================================
# 5. 构建序列
# ======================================================

pivot_entry = df.pivot_table(index="datetime", columns="stationCode", values="entry_interval")
pivot_exit  = df.pivot_table(index="datetime", columns="stationCode", values="exit_interval")
pivot_entry = pivot_entry[station_order]; pivot_exit = pivot_exit[station_order]

time_df = df.drop_duplicates("datetime").sort_values("datetime")
time_data = time_df[time_features].values
external_data = time_df[external_features].values

GAMMA = 0.5

def compute_ema_baseline(pivot, gamma):
    values = pivot.values
    ema = np.zeros_like(values, dtype=np.float64)
    for slot in range(STEPS_PER_DAY):
        indices = np.arange(slot, len(values), STEPS_PER_DAY)
        if len(indices) == 0:
            continue
        slot_vals = values[indices]
        slot_ema = np.zeros_like(slot_vals)
        slot_ema[0] = slot_vals[0]
        for d in range(1, len(slot_vals)):
            slot_ema[d] = gamma * slot_vals[d - 1] + (1 - gamma) * slot_ema[d - 1]
        ema[indices] = slot_ema
    return ema

ema_entry = compute_ema_baseline(pivot_entry, GAMMA)
ema_exit  = compute_ema_baseline(pivot_exit, GAMMA)
print(f"EMA 基线: entry={ema_entry.shape}, exit={ema_exit.shape}")

MAX_SEASONAL_LAG = 7 * STEPS_PER_DAY  # 1008


def build_sequences(start_dt_idx, end_dt_idx):
    X_entry, X_exit, X_seasonal, X_time, X_ext = [], [], [], [], []
    Y_entry_l, Y_exit_l = [], []

    seq_start = max(start_dt_idx, MAX_SEASONAL_LAG)
    seq_end   = end_dt_idx - TIME_STEPS

    for i in range(seq_start, seq_end):
        X_entry.append(pivot_entry.iloc[i:i + TIME_STEPS].values[..., np.newaxis])
        X_exit.append(pivot_exit.iloc[i:i + TIME_STEPS].values[..., np.newaxis])
        X_time.append(time_data[i:i + TIME_STEPS])
        X_ext.append(external_data[i + TIME_STEPS - 1])

        target_idx = i + TIME_STEPS
        seasonal_parts = []
        seasonal_parts.append(ema_entry[target_idx])
        seasonal_parts.append(ema_exit[target_idx])
        seasonal_parts.append(pivot_entry.iloc[target_idx - STEPS_PER_DAY].values)
        seasonal_parts.append(pivot_exit.iloc[target_idx - STEPS_PER_DAY].values)
        seasonal_parts.append(pivot_entry.iloc[target_idx - 7 * STEPS_PER_DAY].values)
        seasonal_parts.append(pivot_exit.iloc[target_idx - 7 * STEPS_PER_DAY].values)

        X_seasonal.append(np.stack(seasonal_parts, axis=-1))

        Y_entry_l.append(pivot_entry.iloc[target_idx].values[..., np.newaxis])
        Y_exit_l.append(pivot_exit.iloc[target_idx].values[..., np.newaxis])

    return (
        np.array(X_entry, dtype=np.float32), np.array(X_exit, dtype=np.float32),
        np.array(X_seasonal, dtype=np.float32), np.array(X_time, dtype=np.float32),
        np.array(X_ext, dtype=np.float32),
        np.concatenate([np.array(Y_entry_l, dtype=np.float32),
                        np.array(Y_exit_l, dtype=np.float32)], axis=-1)
    )


X_entry_train, X_exit_train, X_seas_train, X_time_train, X_ext_train, Y_train = \
    build_sequences(0, train_end_idx)
X_entry_val, X_exit_val, X_seas_val, X_time_val, X_ext_val, Y_val = \
    build_sequences(train_end_idx, val_end_idx)
X_entry_test, X_exit_test, X_seas_test, X_time_test, X_ext_test, Y_test = \
    build_sequences(val_end_idx, n_total)

print(f"X_entry: {X_entry_train.shape}  X_seasonal: {X_seas_train.shape}")
print(f"X_external: {X_ext_train.shape}  Y: {Y_train.shape}")

# ======================================================
# 6. ResGCN + TemporalTransformer
# ======================================================

class ResGCNBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.units = units; self.dropout_rate = dropout_rate

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.W1 = self.add_weight(shape=(in_dim, self.units), initializer='glorot_uniform', trainable=True, name='W1')
        self.W2 = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', trainable=True, name='W2')
        self.bn1 = BatchNormalization(); self.bn2 = BatchNormalization()
        self.dropout = Dropout(self.dropout_rate)
        self.proj = self.add_weight(shape=(in_dim, self.units), initializer='glorot_uniform', trainable=True, name='proj') if in_dim != self.units else None
        super().build(input_shape)

    def call(self, x, training=None):
        A_tf = tf.constant(A_hat, dtype=tf.float32)
        outputs = []
        for t in range(x.shape[1]):
            xt = x[:, t, :, :]
            h = tf.matmul(xt, self.W1); h = tf.matmul(A_tf, h)
            h = self.bn1(h, training=training); h = tf.nn.relu(h)
            h = self.dropout(h, training=training)
            h = tf.matmul(h, self.W2); h = tf.matmul(A_tf, h)
            h = self.bn2(h, training=training)
            h = tf.nn.relu(h + (tf.matmul(xt, self.proj) if self.proj is not None else xt))
            outputs.append(h)
        return tf.stack(outputs, axis=1)


class TemporalTransformer(Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(embed_dim * 2, activation='relu'), Dropout(dropout_rate), Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6); self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate); self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=None):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


# ======================================================
# 7. 构建模型
# ======================================================

entry_input    = Input(shape=(TIME_STEPS, NODES, 1), name="entry_input")
exit_input     = Input(shape=(TIME_STEPS, NODES, 1), name="exit_input")
time_input     = Input(shape=(TIME_STEPS, len(time_features)), name="time_input")
seasonal_input = Input(shape=(NODES, SEASONAL_DIM), name="seasonal_input")
external_input = Input(shape=(len(external_features),), name="external_input")

# ---- 进站空间 ----
entry_gcn = ResGCNBlock(32, 0.1)(entry_input)
entry_gcn = ResGCNBlock(32, 0.1)(entry_gcn)
entry_spatial = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(entry_gcn)
entry_spatial = Reshape((NODES, TIME_STEPS * 32))(entry_spatial)  # (37, 384)
entry_spatial = Dense(128, activation='relu')(entry_spatial)
entry_spatial = Dropout(0.1)(entry_spatial)
entry_spatial = Dense(64, activation='relu')(entry_spatial)

# ---- 出站空间 ----
exit_gcn = ResGCNBlock(32, 0.1)(exit_input)
exit_gcn = ResGCNBlock(32, 0.1)(exit_gcn)
exit_spatial = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(exit_gcn)
exit_spatial = Reshape((NODES, TIME_STEPS * 32))(exit_spatial)  # (37, 384)
exit_spatial = Dense(128, activation='relu')(exit_spatial)
exit_spatial = Dropout(0.1)(exit_spatial)
exit_spatial = Dense(64, activation='relu')(exit_spatial)

# ---- 时间周期 ----
time_embed = Dense(32, activation='relu')(time_input)
time_out = TemporalTransformer(embed_dim=32, num_heads=4, dropout_rate=0.1)(time_embed)
time_pooled = Lambda(lambda x: tf.reduce_mean(x, axis=1))(time_out)
time_dense = Dense(32, activation='relu')(time_pooled)
time_spatial = RepeatVector(NODES)(time_dense)

# ---- 外部因素 ----
ext_h = Dense(16, activation='relu')(external_input)
ext_h = Dropout(0.5)(ext_h)
ext_h = Dense(8, activation='relu')(ext_h)
ext_h = Dropout(0.5)(ext_h)
ext_dense = Dense(8, activation='relu')(ext_h)
ext_spatial = RepeatVector(NODES)(ext_dense)

# ==== ① SeasonalEncoder ====
seas_h = Dense(32, activation='relu')(seasonal_input)
seas_h = Dropout(0.15)(seas_h)
seas_h = Dense(24, activation='relu')(seas_h)
seas_h = Dropout(0.15)(seas_h)
seasonal_baseline = Dense(2, name="seasonal_baseline")(seas_h)

# ---- 融合 ----
# entry(64) + exit(64) + time(32) + ext(8) + seas_h(24) = 192
fusion = Concatenate(axis=-1)([entry_spatial, exit_spatial, time_spatial, ext_spatial, seas_h])
fusion = Dense(128, activation='relu')(fusion)
fusion = Dropout(0.1)(fusion)
fusion = Dense(64, activation='relu')(fusion)
fusion = Dropout(0.1)(fusion)

# ==== 输出 = 季节基线 + 时空残差 ====
residual = Dense(2)(fusion)
output = Add(name="output")([seasonal_baseline, residual])

model = Model(
    inputs=[entry_input, exit_input, time_input, seasonal_input, external_input],
    outputs=output, name="Metro_ST_V9_SeasonalBaseline_10min"
)
model.summary()

# ======================================================
# 8. 训练
# ======================================================

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.TerminateOnNaN()
]

history = model.fit(
    [X_entry_train, X_exit_train, X_time_train, X_seas_train, X_ext_train], Y_train,
    validation_data=([X_entry_val, X_exit_val, X_time_val, X_seas_val, X_ext_val], Y_val),
    epochs=80, batch_size=32, callbacks=callbacks, verbose=1
)

# ======================================================
# 9. 评估
# ======================================================

pred = model.predict(
    [X_entry_test, X_exit_test, X_time_test, X_seas_test, X_ext_test], verbose=1
)
pred_2d = pred.reshape(-1, 2); true_2d = Y_test.reshape(-1, 2)
pred_inverse = scaler_flow.inverse_transform(pred_2d)
true_inverse = scaler_flow.inverse_transform(true_2d)
pred_inverse = np.maximum(pred_inverse, 0)

base_hard_2d = X_seas_test[:, :, :2].reshape(-1, 2)
base_hard_inv = scaler_flow.inverse_transform(base_hard_2d)
base_hard_inv = np.maximum(base_hard_inv, 0)


def evaluate(name, pred_v, true_v):
    rmse = np.sqrt(mean_squared_error(true_v, pred_v))
    mae = mean_absolute_error(true_v, pred_v)
    r2 = r2_score(true_v, pred_v)
    mask = true_v > 1
    mape = np.mean(np.abs((true_v[mask] - pred_v[mask]) / true_v[mask])) * 100 if mask.sum() > 0 else np.nan
    wmape = np.sum(np.abs(true_v - pred_v)) / np.sum(np.abs(true_v)) * 100
    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  MAPE: {mape:.2f}%  WMAPE: {wmape:.2f}%")
    return rmse, mae, r2, mape, wmape


print("\n" + "=" * 60)
print("=====  V9 10min 季节编码基线 + 时空残差 (ES-RNN 范式) =====")
print("=====  基线=SeasonalEncoder, 残差=ResGCN+TemporalTransformer =====")
print("=" * 60)

evaluate("Naive Baseline (纯昨天)", base_hard_inv, true_inverse)
evaluate("Model Prediction", pred_inverse, true_inverse)

entry_true = true_inverse[:, 0]; entry_pred = pred_inverse[:, 0]
exit_true  = true_inverse[:, 1]; exit_pred  = pred_inverse[:, 1]
print("\n=====  分进出站  =====")
evaluate("Entry", entry_pred, entry_true)
evaluate("Exit",  exit_pred,  exit_true)

time_index_arr = time_df["time_index"].values
test_seq_start = max(val_end_idx, MAX_SEASONAL_LAG)
test_target_start = test_seq_start + TIME_STEPS
test_time_idx = time_index_arr[test_target_start : test_target_start + len(Y_test)]
peak_test = ((test_time_idx >= 42) & (test_time_idx <= 54)) | ((test_time_idx >= 102) & (test_time_idx <= 114))
peak_mask_flat = np.repeat(peak_test, NODES)
if peak_mask_flat.sum() > 0 and (~peak_mask_flat).sum() > 0:
    print("\n=====  分时段  =====")
    evaluate("Peak Hours",     pred_inverse[peak_mask_flat],  true_inverse[peak_mask_flat])
    evaluate("Off-Peak Hours", pred_inverse[~peak_mask_flat], true_inverse[~peak_mask_flat])

fig_dir = r"D:\PythonProject4\figures_v9_seasonal_10min"
import os; os.makedirs(fig_dir, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train'); axes[0].plot(history.history['val_loss'], label='Val')
best = np.argmin(history.history['val_loss'])
axes[0].axvline(x=best, color='red', linestyle='--', label=f'Best ({best+1})')
axes[0].set_title('Loss - V9 SeasonalBaseline 10min'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['mae'], label='Train'); axes[1].plot(history.history['val_mae'], label='Val')
axes[1].axvline(x=best, color='red', linestyle='--')
axes[1].set_title('MAE - V9 SeasonalBaseline 10min'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, (label, pv, tv) in enumerate([
    ('All', pred_inverse, true_inverse), ('Entry', entry_pred, entry_true), ('Exit', exit_pred, exit_true)
]):
    ax = axes[idx]
    mask = (tv < np.percentile(tv, 99.5)) & (pv < np.percentile(pv, 99.5))
    t, p = tv[mask], pv[mask]
    ax.hist2d(t, p, bins=100, cmap='Blues', range=[[0, np.percentile(t, 99)], [0, np.percentile(p, 99)]])
    lims = [0, max(t.max(), p.max()) * 1.05]; ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)
    r2 = r2_score(t, p); rmse = np.sqrt(mean_squared_error(t, p))
    ax.text(0.05, 0.92, f'R$^2$={r2:.3f}\nRMSE={rmse:.1f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_title(label); ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'scatter.png'), dpi=150); plt.close()
# ======================================================
# 10. 综合可视化 —— 预测vs实际、残差分析、站点对比等
# ======================================================
exec(compile(open(r"D:\PythonProject4\viz_comprehensive_block.py", encoding='utf-8').read(), 'viz_comprehensive_block.py', 'exec'))

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\94742\Desktop\数据2\metro_flow_5min2.csv")

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " +
    df["hour"].astype(str).astype(str).str.zfill(2) + ":" +
    df["minute"].astype(str).str.zfill(2)
)

df = df.sort_values(["stationCode","datetime"])

result = []

for station, g in df.groupby("stationCode"):

    g = g.set_index("datetime")

    # 补齐5分钟时间
    g = g.resample("5min").asfreq()

    # 补站点信息
    g["stationCode"] = station
    g["stationName"] = g["stationName"].ffill().bfill()
    g["lineName"] = g["lineName"].ffill().bfill()

    # 插值
    g["entry_interval"] = g["entry_interval"].interpolate()
    g["exit_interval"] = g["exit_interval"].interpolate()

    result.append(g)

df_clean = pd.concat(result).reset_index()

# 删除异常跳变
df_clean.loc[df_clean["entry_interval"] > 2000,"entry_interval"] = np.nan
df_clean.loc[df_clean["exit_interval"] > 2000,"exit_interval"] = np.nan

# 再插值
df_clean["entry_interval"] = df_clean["entry_interval"].interpolate()
df_clean["exit_interval"] = df_clean["exit_interval"].interpolate()

# 转整数
df_clean["entry_interval"] = df_clean["entry_interval"].round().astype(int)
df_clean["exit_interval"] = df_clean["exit_interval"].round().astype(int)

# ⭐ 最后计算 total
df_clean["total_interval"] = (
    df_clean["entry_interval"] + df_clean["exit_interval"]
)

# 恢复时间字段
df_clean["date"] = df_clean["datetime"].dt.date
df_clean["hour"] = df_clean["datetime"].dt.hour
df_clean["minute"] = df_clean["datetime"].dt.minute

df_clean.to_csv(
    r"C:\Users\94742\Desktop\数据2\metro_flow_timeseries_ready_v22.csv",
    index=False
)
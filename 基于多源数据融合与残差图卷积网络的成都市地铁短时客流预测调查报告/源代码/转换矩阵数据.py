import pandas as pd

# 读取成都数据
df = pd.read_csv(r"C:\Users\94742\Desktop\数据2\metro_flow_timeseries_ready_v22.csv")

# 时间格式
df["datetime"] = pd.to_datetime(df["datetime"])

# 按时间排序
df = df.sort_values(["datetime", "stationCode"])

# -------------------------
# 生成进站矩阵
# -------------------------
in_matrix = df.pivot(
    index="datetime",
    columns="stationCode",
    values="entry_interval"
)

# -------------------------
# 生成出站矩阵
# -------------------------
out_matrix = df.pivot(
    index="datetime",
    columns="stationCode",
    values="exit_interval"
)

# 缺失填0
in_matrix = in_matrix.fillna(0)
out_matrix = out_matrix.fillna(0)

# 保存
in_matrix.to_csv(
    r"C:\Users\94742\Desktop\数据2\in_5min_chengdu2.csv",
    header=False,
    index=False
)

out_matrix.to_csv(
    r"C:\Users\94742\Desktop\数据2\out_5min_chengdu2.csv",
    header=False,
    index=False
)

print("转换完成")
# 转置（变成：站点 × 时间）
in_matrix = in_matrix.T
out_matrix = out_matrix.T

# 再保存（覆盖或另存）
in_matrix.to_csv(
    r"C:\Users\94742\Desktop\数据2\in_5min_chengdu2.csv",
    header=False,
    index=False
)

out_matrix.to_csv(
    r"C:\Users\94742\Desktop\数据2\out_5min_chengdu2.csv",
    header=False,
    index=False
)

print("已转置为：station × time（5min）")
print(in_matrix.shape)
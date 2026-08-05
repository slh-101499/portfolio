import pandas as pd
import numpy as np

# 1 读取数据
df = pd.read_csv(r"C:\Users\94742\Desktop\数据2\metro_spring_cleaned_FINAL2.csv")

# 2 构造时间列
df["time"] = pd.to_datetime(
    df["date"].astype(str) + " " +
    df["hour"].astype(str) + ":" +
    df["minute"].astype(str)
)

# 3 排序（非常重要）
df = df.sort_values(["stationCode", "date", "time"])

# 4 按站点+日期做差分
df["entry_interval"] = df.groupby(
    ["stationCode", "date"]
)["totalEntryFlow"].diff()

df["exit_interval"] = df.groupby(
    ["stationCode", "date"]
)["totalExitFlow"].diff()

# 5 第一条记录填0
df["entry_interval"] = df["entry_interval"].fillna(0)
df["exit_interval"] = df["exit_interval"].fillna(0)

# 6 处理负值（跨天累计清零）
df.loc[df["entry_interval"] < 0, "entry_interval"] = 0
df.loc[df["exit_interval"] < 0, "exit_interval"] = 0

# 7 计算5分钟总客流
df["total_interval"] = df["entry_interval"] + df["exit_interval"]

# 8 异常值检测（3σ原则）
mean_flow = df["total_interval"].mean()
std_flow = df["total_interval"].std()

upper = mean_flow + 3 * std_flow
lower = mean_flow - 3 * std_flow

df["total_interval"] = np.clip(df["total_interval"], lower, upper)

# 9 删除无用列（可选）
df_clean = df[[
    "stationCode",
    "stationName",
    "lineName",
    "date",
    "hour",
    "minute",
    "entry_interval",
    "exit_interval",
    "total_interval"
]]

# 10 保存结果
df_clean.to_csv(r"C:\Users\94742\Desktop\数据2\metro_flow_5min2.csv", index=False)

print("数据处理完成")
print(df_clean.head())
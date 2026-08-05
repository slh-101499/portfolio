import pandas as pd
import os

# 你的文件夹路径
folder_path = r"C:\Users\94742\Desktop\数据2"

# 读取所有csv
all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dfs = []

for file in all_files:
    file_path = os.path.join(folder_path, file)
    print("读取:", file)

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except:
        df = pd.read_csv(file_path, encoding="gbk")

    dfs.append(df)

# 合并
df_all = pd.concat(dfs, ignore_index=True)

print("合并后总条数:", len(df_all))

# 时间格式
df_all["startTime"] = pd.to_datetime(df_all["startTime"], errors="coerce")

# 去重（核心唯一键）
df_all = df_all.drop_duplicates(
    subset=["stationCode", "lineCode", "startTime"]
)

print("去重后条数:", len(df_all))

# 排序
df_all = df_all.sort_values(
    ["startTime", "lineCode", "stationCode"]
).reset_index(drop=True)

# 保存最终大表
output_path = r"C:\Users\94742\Desktop\数据2\metro_full_merged2.csv"
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

print("最终文件保存成功:", output_path)
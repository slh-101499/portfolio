import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\94742\Desktop\数据2\metro_spring_cleaned_FINAL2.csv")

# 只保留站点信息
stations = df[['stationCode','stationName','lineName']].drop_duplicates()

stations = stations.sort_values('stationCode')

station_list = stations['stationName'].tolist()

station_index = {s:i for i,s in enumerate(station_list)}

N = len(station_list)

A = np.zeros((N,N))

for line,group in stations.groupby("lineName"):

    g = group.sort_values("stationCode")

    s = g["stationName"].tolist()

    for i in range(len(s)-1):

        a = station_index[s[i]]
        b = station_index[s[i+1]]

        A[a,b] = 1
        A[b,a] = 1

np.savetxt(r"C:\Users\94742\Desktop\数据2\adjacency2.csv",A,delimiter=",")

print("邻接矩阵生成完成:",A.shape)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import datetime
import csv
import os
import sys
if 'ipykernel' in sys.modules:
    # Jupyter Notebook
    from tqdm import tqdm_notebook as tqdm
else:
    # ipython, python script, ...
    from tqdm import tqdm

import analysis_helper as analysis

def cut_data(pathlist, range_min, range_max, save_name = "test"):
    data = []
    for path in tqdm(pathlist):   # ファイルを読み込む
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                for row in reader:
                    date = pd.to_datetime(row[1], format='%Y-%m-%d-%H-%M-%S.%f')
                    if date >= range_min and date <= range_max:
                        data.append(row)
            except:
                # print ('failed to load data: ',row)
                pass
    data.sort(key=lambda x: x[1])
    if save_name == "":
        save_name = "{}".format(pd.to_datetime(data[0][1], format='%Y-%m-%d-%H-%M-%S.%f').date())
    with open("/content/drive/Shareddrives/宇宙線探究/女子学院/data/cut_data/{}.dat".format(save_name), "w") as f: # ファイルの保存
        for i, row in enumerate(data):
            s = '\t'.join([str(i)] + row[1:]) + "\n"
            f.write(s)

def check(path):
    data = [[], []]
    if type(path) == list:
        for tmp_path in path:
           tmp_data = analysis.CWData(tmp_path)
           data[0].extend( tmp_data.data["date"] )
           data[1].extend( tmp_data.data["event"] )
    else:
        tmp_data = analysis.CWData(path)
        data[0].extend( tmp_data.data["date"] )
        data[1].extend( tmp_data.data["event"] )

    plt.figure(figsize=(8, 6))
    plt.plot( data[0], data[1], ".")
    plt.xticks(rotation =45)
    plt.show()

    trace = go.Scatter(
        x = data[0],
        y = data[1],
        mode = 'lines',
        marker= dict(
            color= "blue",
            opacity= 1.0
        ),
    )
    plotdata =[trace]
    layout = go.Layout(
        font = dict(       # グローバルのフォント設定
            size = 18,
        ),
        xaxis=dict(
            title='日付',
        ),
        yaxis1=dict(
            title='イベント番号'
        )
    )
    fig = go.Figure(data=plotdata, layout=layout)
    fig.update_layout(margin=dict(t=15, b=20, l=25, r=25))
    fig.show()

if __name__ == '__main__':
    print()
    
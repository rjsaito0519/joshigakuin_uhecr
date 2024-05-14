import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = {
    "adc": [0, 1, 2, 3, 4, 5, 6],
    "time": [10, 12, 112, 23, 34, 45, 56]
}

indices = [i for i in range(len(a["adc"])) if a["adc"][i] > 3]
print(indices)
print([a["time"][i] for i in indices])

class CWData():
    def __init__(self, path):
        self.data ={
            'event':[],
            'date':[],
            'adc':[],
            'temp':[]
        }
        self.path = path
        self.LoadData(path)
    
    def LoadData(self, path):
        df_tsv = pd.read_table(path, header=None)
        tmp_data = df_tsv.values
        self.data["event"] = tmp_data[:, 0]
        self.data["date"]  = pd.to_datetime(tmp_data[:, 1], format='%Y-%m-%d-%H-%M-%S.%f')
        self.data["adc"]   = tmp_data[:, 3]
        self.data["temp"]  = tmp_data[:, 6]

# data = CWData("data/2024-04-23_0.dat")

def count_digit(value):
    return sum(c.isdigit() for c in str(value))

rng = np.random.default_rng()
print(rng.exponential(5))
for _ in range(10):
    n = rng.random()*10**(_)
    print(n)
    s = "{:e}".format(abs(n))
    print(s[-3:], int(s[-3:])//3, (int(s[-3:])))
    d = r"[$\times 10^{" + str(3*int(s[-3:])//3) + r"} \geq $]"
    print(d)
    plt.plot(0, _*1.1)
    plt.text(0, _, d)
    print(r"Rate of Coincidences [$\times 10^{}$]")
plt.show()

print(count_digit(-12345))     # 実行結果 5
print(count_digit(0.12345))    # 実行結果 6
print(count_digit(-3.14))      # 実行結果 3

print(1/60)

a = np.linspace(0, 10, 10)
b = np.linspace(10, 100, 12)
c = np.hstack([a, b, np.nan])
print(np.where(~np.isnan(c), c, 0))
print(np.max(np.where(~np.isnan(c), c, 0)))
print(np.nan / 1000)

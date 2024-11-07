import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('result.csv')
df['degree'] = [100, 200, 400, 800] 
df.set_index('degree', inplace=True, drop=True)
print(df)

for i in range(len(df.columns)):
    x = df.index.values
    y = df.iloc[:, i].values
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    print(f"Slope for column {df.columns[i]}: {slope}")
    
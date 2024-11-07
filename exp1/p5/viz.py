import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data2.csv')
df['degree'] = [10, 20, 40, 80] 
df.set_index('degree', inplace=True, drop=True)
print(df)

for i in range(4):
    plt.clf()
    plt.title(f'{df.columns[i]} vs degree')
    plt.xlabel('degree')
    plt.ylabel(df.columns[i])
    if i == 0:
        plt.ylabel('time[sec]')
    elif i == 1:
        plt.ylabel('loop_count')
    elif i == 2 or i == 3:
        plt.ylabel('relative_err')
    plt.plot(df.index, df.iloc[:, i], label=df.columns[i])
    plt.scatter(df.index, df.iloc[:, i])
    plt.savefig(f'p72_{df.columns[i]}.png')
    
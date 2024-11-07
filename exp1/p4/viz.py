import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ex_data.csv')
df['degree'] = [100, 200, 400, 800] 
df.set_index('degree', inplace=True, drop=True)
print(df)

for i in range(5):
    plt.clf()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{df.columns[i]} vs degree')
    plt.xlabel('degree')
    plt.ylabel(df.columns[i])
    # if i == 0:
    #     plt.ylabel('time[sec]')
    # elif i == 1:
    #     plt.ylabel('loop_count')
    # elif i == 2 or i == 3:
    #     plt.ylabel('relative_err')
    plt.plot(df.index, df.iloc[:, i], label=df.columns[i])
    plt.scatter(df.index, df.iloc[:, i])
    plt.savefig(f'p7_{df.columns[i]}.png')
    
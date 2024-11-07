from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = Path(__file__).resolve().parent / "data"
sys.path.append(Path(__file__).resolve().parent)

from p2 import min_sq_err

def min_wsq_err(x:np.ndarray,phi:np.ndarray,y:np.ndarray,V:np.ndarray)->np.ndarray:
    Q = np.linalg.inv(V)
    
    N = x.shape[0]  # データ数
    n = phi(x,0).shape[1]  # theta_dim

    inv_mat = np.zeros((n,n))
    right_mat = np.zeros((n,1))
    
    for idx in range(N):
        inv_mat += phi(x,idx).T@Q@phi(x,idx)
        right_mat += phi(x,idx).T@Q@y[idx]
    
    theta_hat = np.linalg.inv(inv_mat)@right_mat
    
    return theta_hat

def main():
    # 1. Read data
    data = pd.read_excel(
        str(data_path / "mmse_kadai5.xls"),
        header=None
    )
    data = data.to_numpy()
    
    def phi(x:np.ndarray,idx:int)->np.ndarray:
        return np.array([[1,x[idx]],[1,x[idx]**2]])
    
    V = np.array([
        [100,0],
        [0,1]
    ])
    
    
    deg_list = [2**k for k in np.arange(1,14,1)]
    theta_process = np.ndarray((len(deg_list),2))
    print(theta_process.shape)
    for idx,deg in enumerate(deg_list):
        theta = min_wsq_err(data[:deg,0],phi,data[:deg,1:],V)
        theta_process[idx,0] = theta[0]
        theta_process[idx,1] = theta[1]
    plt.xscale("log")
    plt.plot(deg_list,theta_process[:,0])
    plt.savefig("1.png")
    
    plt.clf()
    plt.plot(deg_list,theta_process[:,1])
    plt.tight_layout()
    plt.savefig("2.png")
    
if __name__ == '__main__':
    main()




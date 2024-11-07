from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = Path(__file__).resolve().parent / "data"
sys.path.append(Path(__file__).resolve().parent)

def phi(x: np.ndarray,idx:int)->np.ndarray:
    return np.ndarray(1,x[idx],x[idx]**2,x[idx]**3).reshape(1,-1)

def min_sq_err(x:np.ndarray,y:np.ndarray,phi:callable):

    N = x.shape[0]  # データ数
    n = phi(x,0).shape[1]  # theta_dim

    inv_mat = np.zeros((n,n))
    right_mat = np.zeros((n,1))
    
    for idx in range(N):
        inv_mat += np.dot(phi(x,idx).T,phi(x,idx))
        right_mat += phi(x,idx).T*y[idx]
    
    theta_hat = np.dot(np.linalg.inv(inv_mat),right_mat)
    
    return theta_hat

def conv_mat(x:np.ndarray,phi: callable,y:np.ndarray,theta: np.ndarray)->np.ndarray:
    N = y.shape[1]

    conv_mat = np.zeros((N,N))
    for idx in range(N):
        conv_mat += (y[idx]-np.dot(phi(x,idx),theta))*(y[idx]-np.dot(phi(x,idx),theta)).T
    
    return conv_mat/(N-theta.shape[0])
    

def main():
    # 1. Read data
    data = pd.read_excel(
        str(data_path / "mmse_kada2.xls"),
        header=None
    )
    data = data.to_numpy()
    deg_list = [2**k for k in np.arange(1,14,1)]
    
    theta_process = np.ndarray((len(deg_list),2))
    print(theta_process.shape)
    for idx,deg in enumerate(deg_list):
        theta,_ = poly_reg(data[:deg])
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
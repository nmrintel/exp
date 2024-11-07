import matplotlib as plt
import numpy as np

from p1 import min_sq_err,conv_mat

def min_wsq_err_core(x:np.ndarray,phi:np.ndarray,y:np.ndarray,V:np.ndarray)->np.ndarray:
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

def min_wsq_err_pipeline(x: np.ndarray,y: np.ndarray,phi:callable):
    theta_hat = min_sq_err(x,)
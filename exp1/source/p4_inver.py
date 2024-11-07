import numpy as np
import sys
from pathlib import Path
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path('.').resolve().parent))
from p2.LU_decompose import LU_decompose,LU_solve

EPS = 10**(-10)
EPS2 = 10**(-5)
LOOP_NUM = 100

def exponentiation(A:np.ndarray)->tuple[float,np.ndarray,float]:
    x:np.ndarray = np.random.rand(A.shape[0])
    x /= np.linalg.norm(x)
    y:np.ndarray = np.zeros(A.shape[0])
    
    mu_old:float = 10
    mu_new:float = 1
    
    conv_cnt = 0
    
    while(np.abs((mu_old-mu_new)/mu_old)> EPS2):
        y = np.matmul(A,x)
        idx_abs_x_max = np.argmax(np.abs(x))
        mu_old = mu_new
        mu_new = y[idx_abs_x_max]/x[idx_abs_x_max]
        x = y/np.linalg.norm(y)
        conv_cnt +=1
    return mu_new,x,conv_cnt

def inverse_iteration(A: np.ndarray,x:np.ndarray ,sigma: float) -> tuple[float, np.ndarray, float]:
    # Aは固有値を実数にするため、対称行列とする
    B = A - sigma * np.identity(A.shape[0])
    P, L, U = LU_decompose(B)

    x /= np.linalg.norm(x)
    y: np.ndarray = np.zeros(A.shape[0])
    
    mu_old: float = 10
    mu_new: float = 1
    
    cnt = 0
    while np.abs((mu_old - mu_new) / mu_old) > EPS:
        y = LU_solve(P, L, U, x)
        idx_abs_x_max = np.argmax(np.abs(x))
        mu_old = mu_new
        mu_new = y[idx_abs_x_max] / x[idx_abs_x_max]
        x = y / np.linalg.norm(y)
        cnt += 1
    
    return 1 / mu_new + sigma, x, cnt

def analysis(degree: int)->list:
    elapsed_time:float = 0.0
    resi_norm:float = 0.0
    rel_err:float = 0.0
    eg_rel_err = 0
    ev_rel_err = 0
    
    cnt_sum = 0
    
    for i in tqdm(range(LOOP_NUM)):

        #乱数行列の初期化
        matrix = np.random.rand(degree, degree)
        matrix *= 2
        matrix -= 1
        matrix += np.transpose(matrix)
        matrix /= 2
        cnt = 0
        cnt2 = 0
        
        
        #自作関数での解法
        start_time = time()
        
        eg_sigma,x,cnt = exponentiation(matrix)
        
        eg,vec,cnt2 = inverse_iteration(matrix,x,eg_sigma)
        end_time = time()
        elapsed_time += (end_time-start_time)
        
        #ライブラリでの解法
        egs_lib,vecs_lib = np.linalg.eig(matrix)
        vecs_lib_ind = np.argmax(np.absolute(egs_lib))
        eg_lib = egs_lib[vecs_lib_ind]
        egv_lib = vecs_lib[:,vecs_lib_ind]
        
        egv_lib /= np.linalg.norm(egv_lib)
    
        if vec[0]<0:
            vec = -vec
        if egv_lib[0]<0:
            egv_lib = -egv_lib
            
        # print(vec)
        # print(egv_lib)
        # print(eg,eg_lib)
        print("")
            
        cnt_sum+=(cnt + cnt2)
        resi_norm += np.linalg.norm(np.subtract(np.matmul(matrix,vec),np.matmul(matrix,egv_lib)))
        eg_rel_err += np.abs((eg-eg_lib)/eg_lib)
        ev_rel_err += np.linalg.norm(np.subtract(vec,egv_lib))/np.linalg.norm(egv_lib)
        
        
    cnt_sum /= LOOP_NUM
    elapsed_time /= LOOP_NUM
    resi_norm /= LOOP_NUM
    eg_rel_err /=LOOP_NUM
    ev_rel_err /= LOOP_NUM
    data = [elapsed_time,resi_norm,eg_rel_err,ev_rel_err,cnt_sum]
    return data

def main():
    
    data = []
    deg_list = [100,200,400,800]
    for i, deg in enumerate(deg_list):
        data.append(analysis(degree=deg))
    
    labels = ["elapsed_time","residual_norm","eigen_value_relative_err","eigen_vector_relative_err","converganece_count"]
    import pandas as pd
    df = pd.DataFrame(data,columns=labels)
    df.to_csv("result_inv.csv")
    for idx,name in enumerate(labels):
        plt.clf() 
        plt.scatter(deg_list,data[:,idx])
        plt.plot(deg_list,data[:,idx])
        plt.title(f"degree vs {name}")
        if(idx==0):
            plt.ylabel("time[sec]")
        elif(idx ==4):
            plt.ylabel("loop_count")
        else:
            plt.ylabel("relative_err")
        plt.xlabel("size of matrix")
        plt.savefig(f"p5_{name}.png")
    

if __name__ == "__main__":
    main()
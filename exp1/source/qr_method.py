import numpy as np
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


import logging

# Update the logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append(str(Path(__file__).resolve().parent))

EPS = 10**-8
EPS2 = 10**-6
LOOP_CNT_SHIFT = 100
LOOP_NUM = 10
logging.basicConfig()

def check_lmat_zero(A: np.ndarray,Thereshold:float)-> bool:
    size:int = A.shape[0]
    
    zero_flag:bool = True
    for i in range(1,size):
        for j in range(i-1):
                if np.abs(A[i, j]/A[i,i]) > Thereshold:
                    zero_flag = False
                    break
    return zero_flag

def check_nrow_zero(A: np.ndarray,Threshold:float)->bool:
    size:int = A.shape[0]
    zero_flag:bool = True
    
    for i in range(size-1):
            if np.abs(A[size-1,i])>Threshold*abs(A[size-1,size-1]):
                zero_flag = False
            break 
    return zero_flag

def qr_decomposition(A_args:np.ndarray)-> tuple[np.ndarray,np.ndarray]:
    A = A_args.copy()
    size = A.shape[0]
    
    R:np.ndarray = np.zeros((size,size))
    Q:np.ndarray = np.ndarray((size,size))
    
    #1列目の正規化
    R[0,0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0]/R[0,0]
    
    for j in range(1,size):
        b:np.ndarray =  A[:,j]
        for i in range(j):
            R[i,j] = np.dot(Q[:,i],b)
            b -= R[i,j]*Q[:,i]
        
        R[j,j] = np.linalg.norm(b)
        Q[:,j] = b/R[j,j]
    
    return Q,R

def qr_method(A: np.ndarray) -> np.ndarray:
    size: int = A.shape[0]
    
    cnt = 0
    
    #狭義下三角成分が0がすべて0になるまで
    while not check_lmat_zero(A,EPS):
        Q, R = qr_decomposition(A)
        A = np.matmul(R, Q)
        cnt += 1

    # 対角成は固有値
    eigen_values = np.ndarray(size)
    for i in range(size):
        eigen_values[i] = A[i, i]

    eigen_values.sort()
    return eigen_values,cnt

def qr_method_quick(A_arg: np.ndarray) -> np.ndarray:
    A = A_arg.copy()
    size: int = A.shape[0]
    
    #対角成分がどれだけshift下か
    eigen_values:np.ndarray = np.zeros(size)
    
    #(size,size)成分に従って、シフト->減次の処理を繰り返しえして、
    #最終的に2×2行列までサイズを小さくしていく
    
    cnt = 0
    while A.shape[0] > 0:
        size = A.shape[0]
        
        while not check_nrow_zero(A,EPS2):
            Q, R = np.linalg.qr(A)
            A = np.matmul(R, Q)
        
        #n行目の変化率がEPS2以下なら、近似できているとしてシフトする
        lambda_n:float = A[size-1,size-1]
        for i in range(size):
            #原点シフト
            A[i,i] -= lambda_n
            #シフトした分だけ固有値の調整
            eigen_values[i] += lambda_n
            
        #n,n成分が確定するまで、繰り返し
        while not check_nrow_zero(A,EPS):
            Q, R = qr_decomposition(A)
            A = np.matmul(R, Q)
        
        eigen_values[size-1] += A[size-1,size-1]
        A = A[:-1,:-1]
        
        cnt += 1
        
    return eigen_values,cnt

def analysis(degree: int)->list:
    elapsed_time:float = 0.0
    resi_norm:float = 0.0
    eg_rel_err_sum = 0
    resi_norm_sum = 0
    
    cnt_sum = 0
    
    for i in tqdm(range(LOOP_NUM)):

        #乱数行列の初期化
        matrix = np.random.rand(degree, degree)
        matrix *= 2
        matrix -= 1
        matrix += np.transpose(matrix)
        matrix /= 2
        cnt = 0
        
        #自作関数での解法
        start_time = time()
        egs, cnt = qr_method_quick(matrix)
        egs = np.sort(egs)
        end_time = time()
        elapsed_time += (end_time-start_time)
        
        #ライブラリでの解法
        egs_lib,egvs_lib = np.linalg.eig(matrix)
        idx_lib = np.argsort(egs_lib)
        eg_lib = egs_lib[idx_lib]
        egv_lib = egvs_lib[:,idx_lib]
        
        egv_lib /= np.linalg.norm(len(egs))
        print((egs-eg_lib)/eg_lib)
        
        resi_norm = 0
        rel_err = 0
        for i in range(degree):
            resi_norm1 = (matrix-egs[i]*np.eye(degree))@egv_lib[:,i]
            if np.linalg.norm(resi_norm1) > resi_norm:
                resi_norm = np.linalg.norm(resi_norm1)
                
            rel_err1 = np.abs((egs[i]-eg_lib[i])/eg_lib[i])
            print(rel_err1)
            if rel_err1 > rel_err:
                rel_err = rel_err1
        
        cnt_sum += cnt
        eg_rel_err_sum += rel_err
        resi_norm_sum += resi_norm
       
        
    cnt_sum /= LOOP_NUM
    elapsed_time /= LOOP_NUM
    eg_rel_err_sum /=LOOP_NUM
    resi_norm_sum /= LOOP_NUM
    data = [elapsed_time,cnt_sum,resi_norm_sum,eg_rel_err_sum]
    return data

def main():
    data = []
    deg_list = [5]
    labels = ["elapsed_time", "cnt", "residual_norm", "eigen_value_relative_err"] 
    print(labels)
    
    for i in deg_list:
        d = analysis(degree=i)
        data.append(d)
    
    data = np.array(data)
    print("-----------------")
    import pandas as pd
    df = pd.DataFrame(data,columns=labels,index=deg_list)
    print(df.head())
    df.to_csv("data2.csv",index=False)
    
    # for idx,name in enumerate(labels):
    #     plt.clf() 
    #     print(idx,name)
    #     y = list(data[:,idx].copy())
       
    #     plt.scatter(deg_list,y)
    #     plt.plot(deg_list,data[:,idx])
    #     plt.title(f"degree vs {name}")
    #     if(idx==0):
    #         plt.ylabel("time[sec]")
    #     elif(idx ==4):
    #         plt.ylabel("loop_count")
    #     else:
    #         plt.ylabel("relative_err")
    #     plt.xlabel("size of matrix")
    #     plt.savefig(f"p5_{name}.png")
    

if __name__ == "__main__":
    main()

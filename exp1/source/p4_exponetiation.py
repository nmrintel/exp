import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path

LOOP_NUM = 100
sys.path.append(str(Path('.').resolve().parent))

EPS = 10**(-10)
EPS2 = 10**(-2)

def exponentiation(A:np.ndarray)->tuple[float,np.ndarray,float]:
    x:np.ndarray = np.random.rand(A.shape[0])
    x /= np.linalg.norm(x)
    y:np.ndarray = np.random.rand(A.shape[0])
    
    mu_old:float = 10
    mu_new:float = 1
    
    conv_cnt = 0
    
    while(np.abs((mu_old-mu_new)/mu_old)> EPS):
        x = y/np.linalg.norm(y)
        y = np.matmul(A,x)
        idx_abs_x_max = np.argmax(np.abs(x))
        mu_old = mu_new
        mu_new = y[idx_abs_x_max]/x[idx_abs_x_max]
        
        conv_cnt +=1
        # if(conv_cnt>20000):
        #     return mu_new,x,conv_cnt
    return mu_new,x,conv_cnt

def analysis(degree: int)->list:
    elapsed_time:float = []
    resi_norm:float = []
    eg_rel_err = []
    ev_rel_err = []
    
    cnt = []
    list = []
    
    for i in tqdm(range(LOOP_NUM)):
        list.append(i)

        #乱数行列の初期化
        matrix = np.random.rand(degree, degree)
        matrix *= 2
        matrix -= 1
        matrix += np.transpose(matrix)
        matrix /= 2
        
        cnt1 = 0
        
        
        #自作関数での解法
        start_time = time()
        eg,vec,cnt1 = exponentiation(matrix)
        end_time = time()
        elapsed_time.append(end_time-start_time)
        
        cnt.append(cnt1)
        
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
            
        # print(eg,eg_lib
        resi_norm.append(np.linalg.norm(np.subtract((vec*eg),np.matmul(matrix,vec))))
        eg_rel_err.append(np.abs((eg-eg_lib)/eg_lib))
        ev_rel_err.append(np.linalg.norm(np.subtract(vec,egv_lib))/np.linalg.norm(egv_lib))
        
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Flatten the axs array for easy iteration
        
    axs = axs.flatten()
        
        # Data to plot
    data_to_plot = [elapsed_time, resi_norm, eg_rel_err, ev_rel_err, cnt]
    titles = ["elapsed_time", "Residual Norm", "Eigenvalue Relative Error", "Eigenvector Relative Error", "Convergence Count"]
    y_labels = ["time[s]", "Residual Norm", "Relative Error", "Relative Error", "Count"]
    
    for i, data in enumerate(data_to_plot):
        ax = axs[i]
        ax.scatter(list,data, label=titles[i])
        ax.set_yscale('log')
        ax.set_title(titles[i])
        ax.set_ylabel(y_labels[i])
        ax.set_xlabel("traial")
        ax.legend()
        
        # Plot the mean line
        mean_value = np.mean(data)
        ax.axhline(y=mean_value, color='r', linestyle='-', label='Mean')
        ax.legend()
        # Hide the last subplot (6th) as we only have 5 plots
    fig.delaxes(axs[-1])
        
    plt.tight_layout()
    plt.savefig(f"{degree}_analysis_plots.png")
    return

def main():
    
    data = []
    deg_list =[100,200,400,800]
    excute_data = np.zeros((len(deg_list),5))
    for i, deg in enumerate(deg_list):
        analysis(degree=deg)
       
if __name__ == "__main__":
    main()
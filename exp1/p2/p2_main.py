import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path('.').resolve().parent))
from LU_decompose import LU_main

LOOP_NUM = 100


def analysis(degree: int)->tuple[float,float,float]:
    elapsed_time:float = []
    resi_norm:float = []
    rel_err:float = []
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    list = []
    
    for i in tqdm(range(LOOP_NUM)):
        list.append(i)

        #乱数行列の初期化
        matrix = np.random.rand(degree,degree)
        b = np.random.rand(degree)
        matrix *= 2
        matrix -= 1
        
        #自作関数での解法
        start_time = time()
        ans:np.ndarray = LU_main(matrix,b)
        end_time = time()
        elapsed_time.append(end_time-start_time)
        
        #ライブラリでの解法
        ans_lib:np.ndarray = np.linalg.solve(matrix,b)
        
        #残差ノルム
        resi_vec = np.subtract(b,np.matmul(matrix,ans))
        resi_norm.append(np.linalg.norm(resi_vec))
        rel_err.append(np.linalg.norm(np.subtract(ans,ans_lib))/np.linalg.norm(ans_lib))
        
    axs[0].scatter(list,elapsed_time )
    axs[0].set_title(" elapsed_time")
    axs[0].set_ylabel("time[sec]")
    axs[0].set_xlabel("traial")
    axs[0].set_yscale('log')
    # Plot average elapsed time with a red line
    avg_elapsed_time = np.mean(elapsed_time)
    axs[0].axhline(y=avg_elapsed_time, color='r', linestyle='-', label=f'Average: {avg_elapsed_time:.2f} sec')
    axs[0].legend()

    # Plot residual norm
    axs[1].scatter(list, resi_norm)
    axs[1].set_title("residual_norm")
    axs[1].set_ylabel("residual_norm")
    axs[1].set_xlabel("traial")
    axs[1].set_yscale('log')

    # Plot average residual norm with a red line
    avg_resi_norm = np.mean(resi_norm)
    axs[1].axhline(y=avg_resi_norm, color='r', linestyle='-', label=f'Average: {avg_resi_norm:.2f}')
    

    # Plot relative error
    axs[2].scatter(list, rel_err)
    axs[2].set_title("relative_error")
    axs[2].set_ylabel("relative_error")
    axs[2].set_xlabel("traial")
    axs[2].set_yscale('log')
    avg_rel_err = np.mean(rel_err)
    axs[2].axhline(y=avg_rel_err, color='r', linestyle='-', label=f'Average: {avg_rel_err:.2f}')
    

    plt.tight_layout()
    plt.savefig(f"{degree}_combined_plot.png")
        
   
    return 

def main():
    
    deg_list = [100,200,400,800]
    excute_data = np.ndarray((len(deg_list),3))
    for idx,deg in enumerate(deg_list):
        analysis(degree=deg)
       

if __name__ == "__main__":
    main()
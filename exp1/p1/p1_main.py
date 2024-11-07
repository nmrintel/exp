import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

LOOP_NUM = 100

def forward_erase(matrix_arg: np.ndarray) -> np.ndarray:
    matrix = matrix_arg.copy()
    n: int = matrix.shape[0]
    for k in range(0, n-1):
        pivot_index: int = np.argmax(np.absolute(matrix[k, k:n])) + k
        if k != pivot_index:
            matrix[[k, pivot_index]] = matrix[[pivot_index, k]]
        
        for i in range(k+1, n):
            try:
                factor = matrix[i, k] / matrix[k, k]
                matrix[i, :] -= factor * matrix[k, :]
            except ZeroDivisionError:
                raise ValueError("zero division")
    
    return matrix


def solve(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    ans_vec = np.zeros(n)
        
    for i in range(n-1, -1, -1):
        ans_vec[i] = (matrix[i, n] - np.sum(matrix[i, i+1:n] * ans_vec[i+1:n])) / matrix[i, i]
        
    return ans_vec

def analysis(degree: int)->tuple[float,float]:
    elapsed_time = []
    resi_norm = []
    rel_err =[]
    resi_vec = []
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot elapsed time
        
    list = []
    
    for i in tqdm(range(LOOP_NUM)):
        list.append(i)
        

        #乱数行列の初期化
        matrix = np.random.rand(degree, degree + 1)
        matrix *= 2
        matrix -= 1
        
        #自作関数での解法
        start_time = time()
        matrix_gauss:np.ndarray = forward_erase(matrix)
        ans:np.ndarray = solve(matrix_gauss)
        end_time = time()
        elapsed_time.append(end_time-start_time)
        

        plt.tight_layout()
        plt.savefig("combined_plot.svg")
        
        #ライブラリでの解法
        ans_lib:np.ndarray = np.linalg.solve(matrix[:,:-1],matrix[:,-1])
        
        #残差ノルム
        resi_vec = (np.subtract(matrix[:,-1],np.matmul(matrix[:,:-1],ans)))
        resi_norm.append(np.linalg.norm(resi_vec))
        rel_err.append(np.linalg.norm(np.subtract(ans,ans_lib))/np.linalg.norm(ans_lib))
        
        # Plot elapsed time
        
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
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

data_path = Path(__file__).resolve().parent / "data"
sys.path.append(Path(__file__).resolve().parent)
logging.basicConfig(level=logging.DEBUG)


def min_sq_err(x: np.ndarray, phi: callable, y: np.ndarray) -> np.ndarray:
    """最小二乗法でのtheta

    Args:
        x (np.ndarray): 入力データ
        phi (callable): 入力データからφを作成
        y (np.ndarray): 出力データ

    Returns:
        np.ndarray: thetaの推定値
    """

    data_size = x.shape[0]
    N = (phi(x, 0)).shape[1]
    # φの返り値の次元が不明なので、0番目のxを用いて特定

    if (len(y.shape) == 1):
        y = y.reshape(-1, 1)

    inv_mat: np.ndarray = np.zeros((N, N))
    right_mat = np.zeros((N, N))

    for row_idx in range(data_size):
        inv_mat += np.matmul(phi(x, row_idx).T, phi(x, row_idx))
        right_mat += np.matmul(phi(x, row_idx).T, y[row_idx])

    theta_hat = np.linalg.inv(inv_mat) @ right_mat
    return theta_hat


def conv_mat(x: np.ndarray, phi: callable, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    data_size = x.shape[0]
    dim_theta = theta.shape[0]

    conv_mat = np.zeros((dim_theta, dim_theta))
    for idx in range(dim_theta):
        conv_mat += (y[idx]-phi(idx)@theta)@(y[idx]-phi(idx)@theta).T

    return conv_mat/(data_size-dim_theta)


def conf_det(data: np.ndarray, theta: np.ndarray):
    N = data.shape[0]
    y_ave = np.mean(data[:, -1])
    def phi(idx): return np.array(data[idx, :-1]).reshape(1, -1)


def main():
    # 1. Read data
    data = pd.read_excel(
        str(data_path / "mmse_kadai1.xls"),
        header=None
    )
    data = data.to_numpy()
    deg_list = [2**k for k in np.arange(1, 14, 1)]

    def phi(x, idx):
        return np.array([x[idx, 0], x[idx, 1]]).reshape(1, -1)

    theta_process = np.ndarray((len(deg_list), 2))

    for idx, deg in enumerate(deg_list):
        theta, _ = min_sq_err(data[:deg, :-1], phi, data[:deg, -1])
        theta_process[idx, 0] = theta[0]
        theta_process[idx, 1] = theta[1]

    plt.xscale("log")
    plt.plot(deg_list, theta_process[:, 0])
    plt.savefig("1.png")

    plt.clf()
    plt.plot(deg_list, theta_process[:, 1])
    plt.tight_layout()
    plt.savefig("2.png")


if __name__ == '__main__':
    main()

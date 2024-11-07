"""
行列の固有値問題: べき乗法:
"""

import numpy as np

A=np.array([[1,2],[3,4]])

x0 = np.array([1,0]); x1 = np.array([0,1])
u = 1.0*x0+2.0*x1 #初期固有ベクトルです。適当です。

rel_eps = 0.0001 # 固有値の収束条件
#クリロフ列生成

rel_delta_u=100.0
while rel_delta_u >= rel_eps :  # メインループ
    uu = u/np.linalg.norm(u) #正規化(ノルムを1にする)
    print("u=",uu)
    
    u = np.dot(A,uu.T)

    eigen_value=np.dot(uu,u)/(np.dot(uu,uu.T))

    print("eigen_value=",eigen_value)
    
    delta_u_vec = uu-u/np.linalg.norm(u)
    abs_delta_u_value= np.linalg.norm(delta_u_vec)
    rel_delta_u=abs_delta_u_value/np.abs(eigen_value) # 繰り返しステップに対する固有値の相対変化量
    print("rel_delta_u_vec = ",rel_delta_u)



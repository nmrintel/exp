import numpy as np

def calc_k(theta_prev:float)-> tuple[float,float]:
    theta_next = 0.9 * theta_prev + np.random.normal(loc=0,scale=1)
    y_next = 2*theta_next + np.random.normal(loc=0,scale=1)
    return theta_next,y_next

def main():
    theta_0 = np.random.normal(loc=3.0,scale=2.0)
    
    
    print(theta_0)
    
    x_k 

if __name__ == "__main__":
    main()
import numpy as np

def LU_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n: int = A.shape[0]  # A size
    P: np.ndarray = np.identity(n)  # permutation array
    L: np.ndarray = np.zeros((n,n))  # lower triangular A
    
    
    for k in range(0, n-1):
        # (1) decide pivot
        pivot_index: int = np.argmax(np.absolute(A[k:n, k])) + k
        
        # (2) swap rows
        A[[k, pivot_index]] = A[[pivot_index, k]]
        P[[k, pivot_index]] = P[[pivot_index, k]]
        L[[k, pivot_index]] = L[[pivot_index, k]]
        
        # Check for zero pivot
        if A[k, k] == 0:
            raise ValueError(f"Zero pivot encountered at index {k}")
        
        # (3) forward erase
        for i in range(k+1, n):
            factor: float = A[i, k] / A[k, k]
            A[i, :] -= factor * A[k, :]
            L[i, k] = factor
    
    # U is the modified A
    U = A
    L += np.identity(n)
    return P, L, U

def LU_solve(P:np.ndarray,L:np.ndarray,U:np.ndarray,b:np.ndarray)->np.ndarray:
    n = L.shape[0]
    x = np.zeros(n)
    y = np.zeros(n)
    b_pi = np.matmul(P,b)
    
    # Forward substitution
    for i in range(n):
        y[i] = b_pi[i] - np.sum(L[i,:i]*y[:i])
        
    # Backward substitution
    for i in range(n-1,-1,-1):
        x[i] = (y[i]-np.sum(U[i,i+1:n]*x[i+1:n]))/U[i,i]
        
    return x

# Test the functJion
def LU_main(A:np.ndarray,b: np.ndarray)->np.ndarray:
    A_copy = A.copy()
    P,L,U = LU_decompose(A_copy)
    x = LU_solve(P,L,U,b)
    return x

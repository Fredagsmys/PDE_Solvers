import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg

def GaussSeidelMatrix(A):
    A = A.tocsr()
    L = spsp.tril(A).tocsc()
    G = splg.inv(L)@(L-A)
    return G
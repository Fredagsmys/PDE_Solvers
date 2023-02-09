import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
import math

######################################################################################
##                                                                                  ##
##  Lab "Introduction to Finite Element Methods", for the course                    ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##  Based on Matlab code in the book The Finite Element Method: Theory,             ##
##  Implementation, and Applications, by Mats G. Larson and Fredrik Bengzon.        ##
##                                                                                  ##
##  Solves the 1D Poisson equation -u''= f, on domain a < x < b using P1 finite     ##
##  elements                                                                        ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python 3.8.16                                                                 ##
##  - Numpy 1.24.1                                                                  ##
##  - Scipy 1.10.0                                                                  ##
##  - Matplotlib 3.6.3                                                              ##
##                                                                                  ##
######################################################################################

def my_stiffness_matrix_assembler(x):
    #
    # Returns the assembled stiffness matrix A.
    # Input is a vector x of node coords.
    #
    N = len(x) - 1                  # number of elements
    A = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    for i in range(N):              # loop over elements
        h = x[i+1] - x[i]           # element length
        A[i, i] += 1/h              # assemble element stiffness
        A[i, i+1] += -1/h
        A[i+1, i] += -1/h
        A[i+1, i+1] += 1/h
    A[0, 0] = 1e+6                 # adjust for BC
    A[N, N] = 1e+6
    return A.tocsr()

def my_load_vector_assembler(x):
    #
    # Returns the assembled load vector b.
    # Input is a vector x of node coords.
    #
    N = len(x) - 1
    B = np.zeros(N+1)
    for i in range(N):
        h = x[i+1] - x[i]
        B[i] = B[i] + f(x[i])*h/2
        B[i+1] = B[i+1] + f(x[i+1])*h/2
    return B

def f(x):
    return 2

def main():
    a = 0                                 # left end point of interval
    b = 1                                 # right
    N1 = 500
    N2 = 1000                               # number of intervals
    h1 = (b-a)/N1                           # mesh size
    x1 = np.arange(a, b, h1)                # node coords
    A1 = my_stiffness_matrix_assembler(x1)
    B1 = my_load_vector_assembler(x1)
    xi1 = splg.spsolve(A1, B1)               # solve system of equations
    h2 = (b-a)/N2                           # mesh size
    x2 = np.arange(a, b, h2)                # node coords
    A2 = my_stiffness_matrix_assembler(x2)
    B2 = my_load_vector_assembler(x2)
    xi2 = splg.spsolve(A2, B2) 

    # plt.plot(x, xi)                       # plot solution
    # plt.xlabel('x')
    x = np.arange(a,b,h1)
    exact = x*(1-x)
    # plt.plot(x,x*(1-x),'g')
    # plt.show()    
    print(np.linalg.norm(exact,2))
    print(np.linalg.norm(xi1,2))
    print(np.linalg.norm(xi2,2))
    error = math.log(-1*(np.linalg.norm(xi2,1)-np.linalg.norm(exact,1))/(np.linalg.norm(xi1,1)-np.linalg.norm(exact,1)),2)
    
    print(error)
if __name__ == '__main__':
    main()  
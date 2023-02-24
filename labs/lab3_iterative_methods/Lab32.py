from laplace_equation import laplace_equation
from time import time

######################################################################################
##                                                                                  ##
##  Lab "Iterative methods", for the course "Scientific computing for PDEs"         ##
##  at Uppsala University.                                                          ##
##  Based on Matlab code given in the course "Scientific Computing 3"               ##
##                                                                                  ##
##  Part 2: Solves the Laplace equation on the unit square.                         ##
##                                                                                  ##
##  N      : number of interior grid points in each dimension                       ##
##  method : 'jacobi', 'gauss-seidel', or 'conjugate-gradient'                      ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##   - Python 3.8.16                                                                ##
##   - Numpy 1.24.1                                                                 ##
##   - Scipy 1.10.0                                                                 ##
##   - Matplotlib 3.6.3                                                             ##
##                                                                                  ##
######################################################################################

def main():
    N = 100
    ts = time()
    laplace_equation(N, 'jacobi',show_plots=False)
    print(f"j,acobi took: {time()-ts} seconds")
    ts = time()
    laplace_equation(N, 'gauss-seidel',show_plots=False)
    print(f"gauss-seidel took: {time()-ts} seconds")
    ts = time()
    laplace_equation(N, 'conjugate-gradient',show_plots=False)
    print(f"conjugate gradient took: {time()-ts} seconds")
    

if __name__ == '__main__':
    main()
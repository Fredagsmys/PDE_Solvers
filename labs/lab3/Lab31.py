import matplotlib.pyplot as plt
from soapfilm import soapfilm
import scipy.sparse.linalg as spsplg

######################################################################################
##                                                                                  ##
##  Lab "Iterative methods", for the course "Scientific computing for PDEs"         ##
##  at Uppsala University.                                                          ##
##  Based on Matlab code given in the course "Scientific Computing 3"               ##
##                                                                                  ##
##  Part 1: Solves the soap film problem on the unit square.                        ##
##                                                                                  ##
##  Input: Nx, Ny the number of interior points in the x- and y-directions.         ##
##                                                                                  ##
##  Output: A the coefficient matrix for the linear system                          ##
##          b the right hand side of the linear system                              ##
##          x, y the coordinates of the grid                                        ##
##          u the solution.                                                         ##
##                                                                                  ##
##  If the solution is computed it is also plotted.                                 ##
##  Put False to the third argument soapfilm(Nx, Ny, False) to hide plots           ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##   - Python 3.8.16                                                                ##
##   - Numpy 1.24.1                                                                 ##
##   - Scipy 1.10.0                                                                 ##
##   - Matplotlib 3.6.3                                                             ##
##                                                                                  ##
######################################################################################
    
def main():
    Nx = 6
    Ny = 6
    plot_solution = True
    A, _, _, _, _ = soapfilm(Nx, Ny, plot_solution)
    plt.spy(A)
    plt.show()


def timed_runs():
    plot_solution = False
    Ns = [100, 200, 400]
    for Nx in Ns:
        for Ny in Ns:
            _, _, _, _, _  = soapfilm(Nx, Ny, plot_solution)

if __name__ == '__main__':
    main()
    #timed_runs()

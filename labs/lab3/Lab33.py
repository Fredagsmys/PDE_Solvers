from JacobiMatrix import JacobiMatrix
from soapfilm import soapfilm
from JacobiMatrix import JacobiMatrix
from GaussSeidelMatrix import GaussSeidelMatrix
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
#plt.style.use('seaborn-v0_8-whitegrid')

######################################################################################
##                                                                                  ##
##  Lab "Iterative methods", for the course "Scientific computing for PDEs"         ##
##  at Uppsala University.                                                          ##
##  Based on Matlab code given in the course "Scientific Computing 3"               ##
##                                                                                  ##
##  Part 3: Convergence of iterative methods                                        ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##   - Python 3.8.16                                                                ##
##   - Numpy 1.24.1                                                                 ##
##   - Scipy 1.10.0                                                                 ##
##   - Matplotlib 3.6.3                                                             ##
##                                                                                  ##
######################################################################################

def plot_eigenvalues(G, matrix_name):
    eigenvalues_G, _ = splg.eigs(G)
    spectral_radius_G = max(abs(eigenvalues_G))
    print(f'Spectral radius of the {matrix_name} iteration matrix is: {spectral_radius_G:.5f}')
    plt.plot(eigenvalues_G.real, eigenvalues_G.imag, 'b.')
    plt.axhline(0, color='black', linewidth=.5)
    plt.axvline(0, color='black', linewidth=.5)
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.title(f'Eigenvalues of the {matrix_name} matrix\nSpectral radius: {spectral_radius_G:.5f}')
    plt.show()

def main():
    Nx = 16
    Ny = 16
    A, _, _, _, _ = soapfilm(Nx, Ny, False)

    GJ = JacobiMatrix(A)
    plot_eigenvalues(GJ, "Jacobi")

    GGS = GaussSeidelMatrix(A)
    plot_eigenvalues(GGS, "Gauss-Seidel")

if __name__ == '__main__':
    main()
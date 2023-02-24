import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def soapfilm(Nx, Ny, show_plots = True):
    #
    # The domain is the unit square
    #
    Lx = 1
    Ly = 1
    #
    # Discretize in the x and y-directions. Nx, Ny interior points
    #
    hx = Lx/(Nx+1)
    hy = Ly/(Ny+1)
    x = np.arange(0, Lx+hx/2, hx)
    y = np.arange(0, Ly+hy/2, hy)
    #
    # Use D+D- in both directions. Order unknowns row by row.
    # Multiply by hx^2. Stencil:
    #
    #      (hx/hy)^2
    # 1 -2(1+(hx/hy)^2) 1
    #      (hx/hy)^2
    #
    # Diagonal block, size Nx x Nx
    #
    gamma = (hx/hy)**2
    D0 = -2*(1+gamma)*spsp.identity(Nx).todok()
    for j in range(Nx-1):
        D0[j,j+1] = 1
        D0[j+1,j] = 1
    #
    # Upper and lower diagonal blocks
    #
    D1 = gamma*spsp.identity(Nx)
    #
    # Fill the matrix with the blocks Ny x Ny blocks
    # Sparse storage. Use a Kronecker product. 
    #
    K0 = spsp.identity(Ny)
    K1 = spsp.dok_matrix((Ny, Ny))
    for j in range(Ny-1):
        K1[j,j+1] = 1
        K1[j+1,j] = 1
    A = (spsp.kron(K0,D0) + spsp.kron(K1,D1)).tocsr()
    #
    # Compute the right hand side b
    #
    b = np.zeros(Nx*Ny)
    #
    # First row, y=0 comes in
    #
    b[:Nx] = -gamma**2*g(x[1:-1],0)
    #
    # Last row, y=1 comes in
    #
    b[-Nx:] = -gamma**2*g(x[1:-1],1)
    #
    # Every row, x=0, x=1 comes in for first and last position
    #
    for r in range(Ny):
        b[r*Nx] -= g(0,y[r+1])
        b[(r+1)*Nx-1] -= g(1,y[r+1])

    # Solve system, and time it
    start_time = time.time()
    u = splg.spsolve(A, b)
    time_to_solve = time.time() - start_time
    print(f'Time to solve system with Nx={Nx}, Ny={Ny}: {time_to_solve:.2e} s.')

    #
    # Plot the solution
    #
    if show_plots:
        plotsol(x, y, u)
    
    return A, b, x, y, u

def g(x,y):
    #
    # Different values for the four sides
    #
    if np.isscalar(x):
        if x == 0:
            g = abs(np.cos(2*np.pi*y) + 0.5)
        elif x == 1:
            g = abs(np.cos(2*np.pi*y) + 0.5)
    elif np.isscalar(y):
        if y == 0:
            g = abs(np.cos(2*np.pi*x) + 0.5)
        elif y == 1:
            g = abs(np.cos(2*np.pi*x) + 0.5)
    return g

def plotsol(x, y, u):
    #
    # Row by row ordering --> matrix u
    #
    u = np.reshape(u, (len(y)-2, len(x)-2))
    #
    # Insert the boundary values given by the function g
    #
    u = np.concatenate((g(0,y[1:-1])[:,None], u, g(1,y[1:-1])[:,None]), axis=1)
    u = np.concatenate((g(x,0)[None,:], u, g(x,1)[None,:]), axis=0)

    fig = plt.figure()

    ax2D = fig.add_subplot(121, aspect='equal')
    ax2D.contourf(x, y, u)
    ax2D.set_xlabel('x')
    ax2D.set_ylabel('y')
    ax2D.set_title('2D plot')
    
    ax3D = fig.add_subplot(122, projection='3d', aspect='auto')
    X, Y = np.meshgrid(x, y)
    ax3D.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
    ax3D.set_title('3D plot')
    plt.show()
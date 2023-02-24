import numpy as np
import numpy.linalg as nplg
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def laplace_equation(N, method, show_plots = True):
    #
    # N      : number of interior grid points in each dimension
    # method : 'jacobi', 'gauss-seidel', or 'conjugate-gradient'
    #
    maxiter = 100
    Nx = N
    Ny = N
    # The domain is the unit square
    Lx = 1
    Ly = 1
    # Discretize in the x and y-directions. Nx,Ny interior points
    hx = Lx/(Nx+1)
    hy = Ly/(Ny+1)
    x = np.arange(0,Lx+hx/2,hx)
    y = np.arange(0,Ly+hy/2,hy)
    #
    # Use D+D- in both directions. Order unknowns row by row.
    # Multiply by hx^2. Stencil:
    #
    #      (hx/hy)^2
    # 1 -2(1+(hx/hy)^2) 1
    #      (hx/hy)^2
    #
    # Diagonal block, size Nx x Nx
    gamma = (hx/hy)**2
    D0 = -2*(1+gamma)*spsp.identity(Nx).todok()
    for j in range(0,Nx-1):
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
    for j in range(0, Ny-1):
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
    for r in range(0, Ny):
        b[r*Nx] -= g(0,y[r+1])
        b[(r+1)*Nx-1] -= g(1,y[r+1])

    # Now we have A and b
    u = np.zeros(Nx*Ny)
    method = method.lower()
    if method == 'jacobi' or method == 'j':
        print('=== Jacobi ===')
        Dinv = spsp.diags(1/A.diagonal())
        L = spsp.tril(A,-1)
        U = spsp.triu(A, 1)
        for iter in range(maxiter):
            print(f'Iteration Count: {iter+1:5d}, Residual: {nplg.norm(A@u-b):5.3e}')
            u = Dinv@(b-(L+U)@u)
    elif method == 'gauss-seidel' or method == 'gs':
        print('=== Gauss-Seidel ===')
        D = spsp.diags(A.diagonal())
        L = spsp.tril(A,-1)
        U = spsp.triu(A, 1)
        for iter in range(maxiter):
            print(f'Iteration Count: {iter+1:5d}, Residual: {nplg.norm(A@u-b):5.3e}')
            v = b - U@u
            u = splg.spsolve_triangular(D+L, v, lower=True)
    elif method == 'conjugate-gradient' or method == 'cg':
        print('=== Conjugate-Gradient ===')
        r = np.copy(b)
        rho = nplg.norm(r)**2
        p = np.copy(r)
        for iter in range(maxiter):
            print(f'Iteration Count: {iter+1:5d}, Residual: {nplg.norm(A@u-b):5.3e}')
            w = A@p
            if abs(np.dot(p,w)) == 0:
                break
            alpha = rho/np.dot(p,w)
            u += alpha*p
            r -= alpha*w
            rho_prev = rho
            rho = nplg.norm(r)**2
            p = r + (rho/rho_prev)*p
    else:
        print(f'Unknown method {method}')

    if show_plots:
        plotsol(x, y, u)

def g(x,y):
    #
    # Different values for the four sides
    #
    if np.isscalar(x):
        if x == 0:
            g = y
        elif x == 1:
            g = y
    elif np.isscalar(y):
        if y == 0:
            g = np.sin(2*np.pi*x)
        elif y == 1:
            g = np.cos(2*np.pi*x)
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
import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
import scipy.sparse as spsp
from scipy.sparse import kron, csc_matrix, eye, vstack, bmat
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import math

import operators as ops
import matplotlib.pyplot as plt
import time
import rungekutta4 as rk4
from matplotlib import cm
from time import time
c = 1 # wave speed
T = 3 # end time
xl = -1 # left boundary
xr = 1 # right boundary
yl = -1/2
yr = 1/2
Lx = xr - xl # domain length
Ly = yr - yl
k = 2*np.pi
def run_simulation(mx, my, show_animation=True):
    
    # Space discretization
    hx = Lx/(mx-1)
    hy = Ly/(my-1)
    
    eyex = np.identity(mx)
    eyey = np.identity(my)
    
    xvec = np.linspace(xl, xr, mx)
    yvec = np.linspace(yl, yr, my)
    X,Y = np.meshgrid(xvec,yvec)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    H,HIx,D1,D2x,e_lx,e_rx,d1_lx,d1_rx = ops.sbp_cent_2nd(mx,hx)
    H,HIy,D1,D2y,e_ly,e_ry,d1_ly,d1_ry = ops.sbp_cent_2nd(my,hy)

    tauL = c**2
    tauR = -c**2

    D2xx = c**2*D2x + tauL*HIx@e_lx.T@d1_lx + tauR*HIx@e_rx.T@d1_rx
    D2yy = c**2*D2y + tauL*HIy@e_ly.T@d1_ly + tauR*HIy@e_ry.T@d1_ry
    
    D = kron(eyey,D2xx) + kron(D2yy,eyex) 

    # Define right-hand-side function
    def rhs(u):
        HL = np.array([u[1],D@u[0]])
        return HL
    # Time discretization
    ht_try = 0.1*hx/c
    mt = int(np.ceil(T/ht_try+1))
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0

    u = np.zeros((mx*my))
    u_t = np.zeros((mx*my))
    for y in range(my):
        for x in range(mx):
            u[y*mx + x] = np.exp((-(hx*x-1)**2-(hy*y-1/2)**2)/0.05**2)
            # u[y*mx + x] = np.cos(2*np.pi*(hx*x-1))*np.cos(2*np.pi*(hy*y-1))*np.cos(c*math.sqrt((2*np.pi*(hx*x-1))**2 + (2*np.pi*(hy*y-1))**2)*0)
    
    w = np.array([u,u_t])
    
    # Initialize plot for animation
    if show_animation:
        ax = plt.axes(projection='3d')
    # Loop over all time steps
    for tidx in range(mt):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)
        # Update plot every 50th time step
        if tidx % 10 == 0 and show_animation: 
            solution = np.reshape(w[0],(my,mx))
            ax.clear()
            ax.plot_surface(X, Y, solution, cmap='cool')
            ax.set_title(f't = {t:.2f}')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1/2,1/2)
            ax.set_zlim(0,1.1)
            plt.pause(1e-8)
            
    # Close figure window
    if show_animation:
        plt.close()

    return w, T, X, Y, hx, hy, c

def plot_final_solution(u, X, Y):
    ax = plt.axes(projection='3d')
    ax.set_ylim(-1/2,1/2)
    ax.set_xlim(-1,1)
    ax.set_zlim(0,1)
    ax.plot_surface(X, Y, u, cmap="cool")
    plt.show()
    plt.pause(3)
    
def main():
    my = 100
    mx = 200
    tstart = time()
    u, T, X, Y, hx, hy, c = run_simulation(mx=mx, my=my, show_animation=False)  
    tend = time()
    print(tend-tstart)
    solution = np.reshape(u[0],(my,mx))
    plot_final_solution(solution,X,Y)


if __name__ == '__main__':
    main()
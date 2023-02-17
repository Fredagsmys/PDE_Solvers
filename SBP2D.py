import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
import scipy.sparse as spsp

import operators as ops
import matplotlib.pyplot as plt
import time
import rungekutta4 as rk4


# Model parameters
c = 3 # wave speed
T = np.pi # end time
xl = -1 # left boundary
xr = 1 # right boundary
yl = -1/2
yr = 1/2

L = xr - xl # domain length
k = 2*np.pi
def run_simulation(mx, my, show_animation=True):
    """Solves the advection equation using finite differences
    and Runge-Kutta 4.
    
    Method parameters: 
    mx:     Number of grid points, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """
    # Space discretization
    hx = (xr - xl)/(mx-1)
    hy = (yr - yl)/(my-1)
    
    eyex = np.identity(mx)
    eyey = np.identity(my)

    xvec = np.linspace(xl, xr-hx, mx)
    yvec = np.linspace(yl, yr-hy, my)

    # _, _, D1 = ops.periodic_expl(mx, hx, order)
    H,HIx,D1,D2x,e_lx,e_rx,d1_lx,d1_rx = ops.sbp_cent_6th(mx,hx)
    H,HIy,D1,D2y,e_ly,e_ry,d1_ly,d1_ry = ops.sbp_cent_6th(my,hy)

    D = np.kron(eyey,D2x) + np.kron(D2y,eyex)   

    # Define right-hand-side function
    def rhs(u):
        HL = np.array([u[1],D@u[0]])
        
        # print(HL)
        return HL
    # Time discretization
    ht_try = 0.1*hx/c
    mt = int(np.ceil(T/ht_try)) # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0

    u = np.zeros((mx*my))
    u_t = u
    for y in range(my):
        for x in range(mx):
            u[y*mx + x] = np.exp((-x**2-y**2)/0.05**2)
    
    
    
    w = np.array([u,u_t])

    # Initialize plot for animation
    if show_animation:
        fig, ax = plt.subplots()
        [line] = ax.plot(xvec, w[0], label='Approximation')
        ax.set_xlim([xl, xr-hx])
        ax.set_ylim([-1, 1.2])
        title = plt.title(f't = {0:.2f}')
        plt.draw()
        plt.pause(1)

    # Loop over all time steps
    for tidx in range(mt):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)
        # Update plot every 50th time step
        if tidx % 1 == 0 and show_animation: 
            line.set_ydata(w[0])
            title.set_text(f't = {t:.2f}')
            plt.draw()
            plt.pause(1e-8)

    # Close figure window
    if show_animation:
        plt.close()

    return w, T, xvec, yvec, hx, hy, L, c

def exact_solution(t, xvec, L, c):
    # T1 = L/c  # Time for one lap
    # t_eff = (t/T1 - np.floor(t/T1))*T1  # "Effective" time, using periodicity
    # u_exact = f(xvec - c*t_eff)

    
    u_exact = np.cos(k*xvec)*np.cos(c*k*T)
    return u_exact

def l2_norm(vec, h):
    return np.sqrt(h)*np.sqrt(np.sum(vec**2))

def compute_error(u, u_exact, hx):
    """Compute discrete l2 error"""
    error_vec = u - u_exact
    relative_l2_error = l2_norm(error_vec, hx)/l2_norm(u_exact, hx)
    return relative_l2_error
    # error = l2_norm(error_vec,hx)
    # return error

def plot_final_solution(u, u_exact, xvec, T):
    fig, ax = plt.subplots()
    ax.plot(xvec, u, label='Approximation')
    plt.plot(xvec, u_exact, 'r--', label='Exact')
    ax.set_xlim([xvec[0], xvec[-1]])
    ax.set_ylim([-1, 1.2])
    plt.title(f't = {T:.2f}')
    plt.legend()
    plt.show()

def main():
    mx = 200
    my = 100
    fig = plt.figure()
    ax = plt.axes(projection='3d') 
    u, T, xvec, yvec, hx, hy, L, c = run_simulation(mx=mx, my=my, show_animation=False)

    
    

    plt.savefig('meth.png')
    plt.show()

if __name__ == '__main__':
    main()    
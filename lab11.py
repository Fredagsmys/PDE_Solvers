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
L = xr - xl # domain length
k = 2*np.pi
tauL = c**2
tauR = -c**2
def run_simulation(mx, method, show_animation=True):
    """Solves the advection equation using finite differences
    and Runge-Kutta 4.
    
    Method parameters: 
    mx:     Number of grid points, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """
    # Space discretization
    hx = (xr - xl)/(mx-1)
    xvec = np.linspace(xl, xr, mx) # periodic, u(xl) = u(xr)
    # xvec = xvec.reshape(mx,1)
    # _, _, D1 = ops.periodic_expl(mx, hx, order)
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = method(mx,hx)

    # print(f"e_l{np.array(e_l.toarray()[0])}")
    e_l = np.array(e_l.toarray())
    e_r = np.array(e_r.toarray())
    d1_l = np.array(d1_l.toarray())
    d1_r = np.array(d1_r.toarray())
    HI = np.array(HI.toarray())
    D2 = np.array(D2.toarray())
    D = c**2*D2 + tauL*HI@e_l.T@d1_l + tauR*HI@e_r.T@d1_r

    # Define right-hand-side function
    def rhs(u):
        res = np.array([u[1],D.astype(float)@u[0].astype(float)])
        #w_t = A*w
        # print(res)
        return res
    # Time discretization
    ht_try = 0.1*hx/c
    mt = int(np.ceil(T/ht_try)+1) # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0
    phi = np.cos(k*xvec)
    phi_t = np.zeros(np.shape(xvec))
    w = np.array([phi,phi_t])

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
    for tidx in range(mt-1):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)
        if tidx % 1 == 0 and show_animation: 
            line.set_ydata(w[0])
            title.set_text(f't = {t:.2f}')
            plt.draw()
            plt.pause(1e-8)

    # Close figure window
    if show_animation:
        plt.close()

    return w, T, xvec, hx, L, c

def exact_solution(t, xvec, L, c):

    
    u_exact = np.cos(k*xvec)*np.cos(c*k*T)
    return u_exact

def l2_norm(vec, h): 
    return np.sqrt(h)*np.sqrt(np.sum(vec**2))

def compute_error(u, u_exact, hx):
    """Compute discrete l2 error"""
    error_vec = u - u_exact
    # relative_l2_error = l2_norm(error_vec, hx)/l2_norm(u_exact, hx)
    error = l2_norm(error_vec,hx)
    return error 
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
    ms = np.array([25,50,100,200])
    hs = (xr - xl)/(ms-1)
    
    # ms = np.array([800])
    methods = np.array([ops.sbp_cent_2nd, ops.sbp_cent_4th, ops.sbp_cent_6th])
    # methods = np.array([ops.sbp_cent_6th])
    errors = np.empty((methods.shape[0],ms.shape[0]))
    
    for i,meth in enumerate(methods):
        for j,m in enumerate(ms):
            u, T, xvec, hx, L, c = run_simulation(mx=m, show_animation=False, method=meth)
            u_exact = exact_solution(T, xvec, L, c)
            error = compute_error(u[0], u_exact, hx)
            errors[i][j] = error
            print(f'SBP order :{str((1+i)*2)}, L2-error m={m}: {error:.2e}')
            # plot_final_solution(u[0], u_exact, xvec, T)
    plt.loglog(hs,errors[0],label="Order 2")
    plt.loglog(hs,errors[1],label="Order 4")
    plt.loglog(hs,errors[2],label="Order 6")
    print(errors)
    plt.grid(visible=True)
    plt.xlabel('Grid spacing')
    plt.ylabel('l2 error')
    plt.legend()
    plt.savefig('meth.png')
    plt.show()

if __name__ == '__main__':
    main()    
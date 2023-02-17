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
def run_simulation(mx=100, method=ops.sbp_cent_6th, show_animation=True):
    """Solves the advection equation using finite differences
    and Runge-Kutta 4.
    
    Method parameters: 
    mx:     Number of grid points, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """
    # Space discretization
    hx = (xr - xl)/(mx-1)
    xvec = np.linspace(xl, xr-hx, mx) # periodic, u(xl) = u(xr)
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
        res = np.array([u[1],D@u[0]])
        # print(res)
        return res
    # Time discretization
    ht_try = 0.01*hx/c
    mt = int(np.ceil(T/ht_try)) # round up so that (mt-1)*ht = T
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
        # Update plot every 50th time step
        if tidx % 20 == 0 and show_animation: 
            line.set_ydata(w[0])
            title.set_text(f't = {t:.2f}')
            plt.draw()
            plt.pause(1e-8)

    # Close figure window
    if show_animation:
        plt.close()

    return w, T, xvec, hx, L, c

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
    ms = np.array([25,50,100,200,400])
    # ms = [200]
    
    methods = np.array([ops.sbp_cent_2nd, ops.sbp_cent_4th, ops.sbp_cent_6th])
    errors = np.empty((methods.shape[0],ms.shape[0]))
    
    
    print(errors)
    for i,meth in enumerate(methods):
        for j,m in enumerate(ms):
            u, T, xvec, hx, L, c = run_simulation(mx=m, show_animation=False, method=meth)
            u_exact = exact_solution(T, xvec, L, c)
            error = compute_error(u[0], u_exact, hx)
            errors[i][j] = error
            print(f'order :{str(meth)}, L2-error m={m}: {error:.2e}')
    # plt.plot(np.log10(ms), np.log10(errors[0]))
    # plt.plot(np.log10(ms), np.log10(errors[1]))
    # plt.plot(np.log10(ms), np.log10(errors[2]))
    plt.plot(ms, errors[0])
    plt.plot(ms, errors[1])
    plt.plot(ms, errors[2])

    print(errors)
    plt.show()
    plt.savefig('meth.png')
    print(errors)

if __name__ == '__main__':
    main()    


######################################################################################
##                                                                                  ##
##  Lab "Introduction to Finite Difference Methods", part 3, for course             ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##  Updated by Martin Almquist, January 2023.                                       ##
##  Based on Matlab code written by Ken Mattsson in June 2022.                      ##
##                                                                                  ##
##  Solves the 1D Maxwell's equations with constant coefficients using              ##
##  summation-by-parts finite differences and the projection method to              ##
##  impose the boundary conditions. Demonstrates the behavior with 4                ##
##  different types of homogeneous boundary conditions.                             ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt
import rungekutta4 as rk4

# Method parameters
# Number of grid points, integer > 15
mx = 201 

# Order of accuracy: 2, 4, or 6.
order = 6

# Type of boundary condition.
# 1 - Electric field at both boundaries
# 2 - Magnetic field at both boundaries
# 3 - Linear combination E + beta*H and E - beta*H at left and right boundary, respectively
# 4 - Both electric and magnetic field at both boundaries
bc_type = 4

# If bc_type 3, specify beta
beta = -30 

# Model parameters
T = 3.6 # end time

# Domain boundaries
xl = -1
xr = 1

# Material properties
mu = 1
eps = 1

A = csc_matrix([[0,1],[1,0]])
C = csc_matrix([[eps,0],[0,mu]])

# Wave speed
c = 1/sqrt(eps*mu)

# Space discretization
N = 2*mx # number of degrees of freedom

hx = (xr - xl)/(mx-1)
xvec = np.linspace(xl,xr,mx)

if order == 2:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(mx,hx)
elif order == 4:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(mx,hx)
elif order == 6:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(mx,hx)
else:
    raise NotImplementedError('Order not implemented.')

e1 = csc_matrix([1,0])
e2 = csc_matrix([0,1])
I_N = eye(N)
I_m = eye(mx)
I_2 = eye(2)
H_bar = kron(I_2,H)
HI_bar = kron(I_2,HI)

def norm(v):
    return np.sqrt(hx)*np.sqrt(np.sum(v**2))

if bc_type == 1:
    L = vstack((kron(e1,e_l),kron(e1,e_r)),format="csc")
elif bc_type == 2:
    L = vstack((kron(e2,e_l),kron(e2,e_r)),format="csc")
elif bc_type == 3:
    L = vstack((kron(e1 + beta*e2,e_l),kron(e1 - beta*e2,e_r)),format="csc")
elif bc_type == 4:
    L = vstack((kron(e1,e_l),kron(e1,e_r),kron(e2,e_l),kron(e2,e_r)),format="csc")
else:
    raise NotImplementedError('Boundary condition type not implemented.')

# Construct RHS matrix using the projection method
P = I_N - HI_bar@L.T@inv(L@HI_bar@L.T)@L
CI = kron(inv(C),I_m)
D = P@CI@kron(A,D1)@P

# Initial data
def gauss(x):
    rstar = 0.1
    return np.exp(-(x/rstar)**2)

Ey = -gauss(xvec) - gauss(xvec)
Hz = -gauss(xvec) + gauss(xvec)
v = np.hstack((Ey,Hz))

# Time discretization
CFL = 0.1/c
ht_try = CFL*hx
mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0,T,mt,retstep=True)

def rhs(v):
    return D@v

# Initialize plot
fig, ax = plt.subplots()
Ey = v[0:mx]
Hz = v[mx:]
[line1] = ax.plot(xvec,Ey,label='Electric field')
[line2] = ax.plot(xvec,Hz,label='Magnetic field')
plt.legend()
ax.set_xlim([xl,xr])
ax.set_ylim([-2,2])
title = plt.title(f't = {0:.2f}')
plt.draw()
plt.pause(0.5)

t = 0
# Loop over all time steps
for tidx in range(mt-1):
    # Take one step with the fourth order Runge-Kutta method.
    v, t = rk4.step(rhs, v, t, ht)

    # Update plot every 50th time step
    if tidx % ceil(5*c) == 0 or tidx == mt-2:
        Ey = v[0:mx]
        Hz = v[mx:]
        line1.set_ydata(Ey)
        line2.set_ydata(Hz)
        title.set_text(f't = {t:.2f}')
        plt.draw()
        plt.pause(1e-3)

# Error for bc_type 1
if bc_type == 1:
    tt = 4 - T
    Ey_exact = -gauss(xvec - tt) - gauss(xvec + tt)
    Hz_exact = -gauss(xvec - tt) + gauss(xvec + tt)
    Ey = v[0:mx]
    Hz = v[mx:]
    error_E = norm(Ey_exact - Ey)/norm(Ey_exact)
    error_H = norm(Hz_exact - Hz)/norm(Hz_exact)
    print(f'Relative l2-error in Ey: {error_E:.2e}. Relative l2-error in Hz: {error_H:.2e}')

plt.show()
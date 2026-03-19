# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 20:12:36 2026

@author: hugoj
"""

import numpy as np
import matplotlib.pylab as plt
import configMatplotlib

from rk4 import rk4


# =============================================================================
#                            Question 1
# =============================================================================

########## CALCUL:


k = 1.0
x0 = 1.0
dt = 0.1
tmax = 10.0

temps = np.arange(0, tmax + dt, dt)
n = temps.size

x_euler = np.empty(n)
x = x0

for i in range(n):
    x_euler[i] = x
    v = -k*x
    x = x + v*dt

# solution analytique
x_theorique = x0*np.exp(-k*temps)
diff = x_euler - x_theorique


############## AFFICHAGE:

plt.title("Q1 : evolution de la densité d'un élément radioactif X → Y")
plt.plot(temps, x_euler, label="Euler")
plt.plot(temps, x_theorique, label="theo")
plt.xlabel("t (s)")
plt.ylabel("densité (quantité de matière) ")
plt.legend()
plt.savefig("tp1_q1_x.pdf", bbox_inches='tight')
plt.show()

plt.title("Q1 : evolution de la densité : diff euler vs théorique ")
plt.plot(temps, diff)
plt.xlabel("t (s)")
plt.ylabel("x - x_theo (quantité de matière)")
plt.savefig("tp1_q1_diff.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 2
# =============================================================================



########## CALCUL:

k = 1.0
k2 = 0.1*k
x0 = 1.0
y0 = 0.0

dt = 0.05
tmax = 20.0

temps = np.arange(0, tmax + dt, dt)
n = temps.size

xvals = np.empty(n)
yvals = np.empty(n)

x = x0
y = y0

for i in range(n):
    xvals[i] = x
    yvals[i] = y
    vx = -k*x # dx/dt 
    vy = k*x - k2*y # dy/dt
    x = x + vx*dt
    y = y + vy*dt


############## AFFICHAGE:

plt.title("Q2 : evolution de la densité d'un élément radioactif Y → Z")
plt.plot(temps, xvals, label="x")
plt.plot(temps, yvals, label="y")
plt.xlabel("t (s)")
plt.ylabel("densité (quantité de matière) ")
plt.legend()
plt.savefig("tp1_q2_xy.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 3
# =============================================================================



########## CALCUL:

w0 = 1.0
x0 = 1.0
v0 = 0.0

dt = 0.001
tmax = 200.0

temps = np.arange(0, tmax + dt, dt)
n = temps.size

xvals = np.empty(n)
vvals = np.empty(n)
evals = np.empty(n)

x = x0
v = v0

for i in range(n):
    xvals[i] = x
    vvals[i] = v
    evals[i] = 0.5*(v**2 + (w0*x)**2)
    ax = -w0**2*x # d2x/dt2 = -w0^2x 
    x = x + v*dt #  dx/dt=v
    v = v + ax*dt # dv/dt=-w0^2 x
    


############## AFFICHAGE:

plt.title(f"Q3 : dt={dt}s Evolution de l'amplitude de oscillateur harmonique avec Euler  ")
plt.plot(temps, xvals)
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.savefig("tp1_q3_x_dt0001.pdf", bbox_inches='tight')
plt.show()

plt.title(f"Q3 : dt={dt}s Evolution de la vitesse de oscillateur harmonique avec Euler ")
plt.plot(temps, vvals)
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.savefig("tp1_q3_v_dt0001.pdf", bbox_inches='tight')
plt.show()

plt.title(f"Q3 : dt={dt}s Evolution de l'energie de oscillateur harmonique avec Euler")
plt.plot(temps, evals)
plt.xlabel("t (s)")
plt.ylabel("E(J)")
plt.savefig("tp1_q3_E_dt0001.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 4 et 5
# =============================================================================

########## CALCUL:

def deriv_osc(t, y, params):
    """
    Derivees pour l'oscillateur harmonique
    """
    omega = params
    dy = np.zeros(2)
    dy[0] = y[1]
    dy[1] = -omega**2*y[0]
    return dy


def euler(t, dt, y, deriv, params):
    """
    Un pas de Euler : y(t+dt) = y(t) + dt*y'(t)
    """
    dy = deriv(t, y, params)
    y[:] = y + dt*dy
    return y


omega = 1.0
x0 = 1.0
v0 = 0.0

dt = 0.1
tmax = 100.0

temps = np.arange(0, tmax + dt, dt)
n = temps.size

xTab = np.empty(n)
vTab = np.empty(n)

y = np.empty(2)
y[0] = x0
y[1] = v0

for i in range(n):
    xTab[i] = y[0]
    vTab[i] = y[1]
    t = temps[i]
    y = euler(t, dt, y, deriv_osc, omega)


############## AFFICHAGE:

plt.title("Q5")
plt.plot(temps, xTab, label="x")
plt.plot(temps, vTab, label="v")
plt.xlabel("t")
plt.legend()
plt.savefig("tp1_q5_xv.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 6
# =============================================================================

# y = (x, y, vx, vy)  => n = 4

########## CALCUL:

def deriv_EM(t, y, params):
    """
    Derivees pour une particule dans E et B uniformes
    """
    E, B = params
    dy = np.zeros(4)
    x = y[0]
    y_coordonnee = y[1] # coordonnée y ne sert pas ici ( à calculer les dérivées )
    vx = y[2]
    vy = y[3]
    # equations
    dy[0] = vx
    dy[1] = vy
    dy[2] = E + vy*B
    dy[3] = -vx*B
    return dy


E = 1.0
B = 1.0

dt = 0.01
tmax = 40.0

temps = np.arange(0, tmax + dt, dt)
n = temps.size

xTab = np.empty(n)
yTab = np.empty(n)

y = np.empty(4)
y[0] = 0.0  # x
y[1] = 0.0  # y
y[2] = 0.0  # vx
y[3] = 0.0  # vy

for i in range(n):
    xTab[i] = y[0]
    yTab[i] = y[1]
    t = temps[i]
    y = euler(t, dt, y, deriv_EM, np.array([E, B]))


############## AFFICHAGE:

plt.title("Q6: trajectoire d'une particule dans le plan x -y soumis au champ electromagnétique")
plt.plot(xTab, yTab)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.savefig("tp1_q6_xy.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 7
# =============================================================================

########## CALCUL:

omega = 1.0
x0 = 1.0
v0 = 0.0

dt_list = [0.01, 0.005, 0.001]
tmax = 100.0


############## AFFICHAGE:

plt.title("Q7")
for dt in dt_list:
    temps = np.arange(0, tmax + dt, dt)
    n = temps.size

    xTab = np.empty(n)
    vTab = np.empty(n)

    y = np.empty(2)
    y[0] = x0
    y[1] = v0

    for i in range(n):
        xTab[i] = y[0]
        vTab[i] = y[1]
        t = temps[i]
        y = euler(t, dt, y, deriv_osc, omega)

    x_theorique = x0*np.cos(omega*temps) + (v0/omega)*np.sin(omega*temps)
    plt.plot(temps, xTab - x_theorique, label=f"dt={dt}")

plt.title("Q7: difference entre Euler et valeur théorique par pas de temps pour l'oscillateur harmonique")
plt.xlabel("t (s)")
plt.ylabel("x - x_th (m)")
plt.legend()
plt.savefig("tp1_q7_diff.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 8
# =============================================================================

########## CALCUL:

omega = 1.0
x0 = 1.0
v0 = 0.0

dtList = [0.01, 0.005, 0.001]
tmax = 100.0


############## AFFICHAGE:

plt.title("Q8")
for dt in dtList:
    temps = np.arange(0, tmax + dt, dt)
    n = temps.size

    xTab = np.empty(n)

    y = np.empty(2)
    y[0] = x0
    y[1] = v0

    for i in range(n):
        xTab[i] = y[0]
        t = temps[i]
        y = rk4(t, dt, y, deriv_osc, omega)

    x_theorique = x0*np.cos(omega*temps) + (v0/omega)*np.sin(omega*temps)
    plt.plot(temps, xTab - x_theorique, label=f"dt={dt}")

plt.title("Q8: difference entre Euler et valeur théorique par pas de temps pour l'oscillateur harmonique")
plt.xlabel("t (s)")
plt.ylabel("x - x_th (m)")
plt.legend()
plt.savefig("tp1_q8_diff.pdf", bbox_inches='tight')
plt.show()


# =============================================================================
#                            Question 9
# =============================================================================

import time
from numba import njit


def sol_edo(t, tmax, dt, init, deriv, params):
    """
    Boucle temps (python)
    """
    temps = np.arange(0, tmax, dt)
    nPoints = temps.size
    x_vect = np.empty(nPoints)

    y = np.empty(init.size)
    y[:] = init

    for i in range(nPoints):
        x_vect[i] = y[0]
        t = temps[i]
        y = rk4(t, dt, y, deriv, params)

    return temps, x_vect


@njit
def derivNumba(t, y, params):
    omega = params
    dy = np.zeros(2)
    dy[0] = y[1]
    dy[1] = -omega**2*y[0]
    return dy


@njit
def rk4Numba(x, dx, y, deriv, params):
    ddx = dx/2.
    d1 = deriv(x, y, params)
    yp = y + d1*ddx
    d2 = deriv(x + ddx, yp, params)
    yp = y + d2*ddx
    d3 = deriv(x + ddx, yp, params)
    yp = y + d3*dx
    d4 = deriv(x + dx, yp, params)
    return y + dx*(d1 + 2*d2 + 2*d3 + d4)/6.


@njit
def sol_edo_numba(t, tmax, dt, init, deriv, params):
    temps = np.arange(0, tmax, dt)
    n = temps.size
    x_vect = np.empty(n)
    y = np.empty(init.size)
    y[:] = init

    for i in range(n):
        x_vect[i] = y[0]
        t = temps[i]
        y = rk4Numba(t, dt, y, deriv, params)

    return temps, x_vect


omega = 1.0
t = 0.0
dt = 0.001
tmax = 1000.0
cond_init=np.array([1.0, 0.0])
                   
# on appelle une première fois pour compiler la méthode
sol_edo_numba(t, 1.0, dt, cond_init, derivNumba, omega)

start = time.time()
sol_edo(t, tmax, dt, cond_init, deriv_osc, omega)
end = time.time()
print(f"python:{end-start}")

start = time.time()
sol_edo_numba(t, tmax, dt,cond_init, derivNumba, omega)
end = time.time()
print(f"numba:{end-start}")

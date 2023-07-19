# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			High-power-charging-Fig3(bd)
 *Date:			2023-07-10 16:33:20
 *Description:复现High Power Charging of the Dicke Quantum Batteries的Fig3 (b),(c)
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema

# init variance

N_list = np.array([6, 8, 10, 12])
g_max = 2
N_g = 50
g_list = np.linspace(0.1, g_max, N_g)

M = 20
wc = 1
wb = 1

t_s = 8 # strength coupling
t_w = 80 # weak coupling
N_t_s = 1000
N_t_w = 10000
t_s_list = np.linspace(0, t_s, N_t_s)
t_w_list = np.linspace(0, t_w, N_t_w)


# define the function used for calculating the energy of quantum batteries
def energy_dicke(g, N):
    """
    the Dicke model
    """
    if g<0.1: # weak coupling
        tlist = t_w_list
    else: # strength coupling
        tlist = t_s_list

    j = N / 2
    n = 2 * j + 1

    a = tensor(destroy(int(M)), qeye(int(n)))
    Jm = tensor(qeye(int(M)), jmat(j, '-'))
    Jp = tensor(qeye(int(M)), jmat(j, '+'))
    Jx = tensor(qeye(int(M)), jmat(j, 'x'))
    Jz = tensor(qeye(int(M)), jmat(j, 'z'))

    H1 = wc * a.dag() * a + wb * Jz
    H2 = 2*wc*Jx*(a.dag()+a)
    H = H1 + g * H2

    state_b0 = spin_state(j, -j)
    state_c0 = basis(M, N)
    psi_0 = tensor(state_c0, state_b0)

    ave_Jz_0 = expect(Jz, psi_0)

    result = mesolve(H, psi_0, tlist, [], [Jz])
    ave_Jz_t = np.array(result.expect[0])
    ind = argrelextrema(ave_Jz_t,np.greater)[0][0]
    return (ave_Jz_t[ind] - ave_Jz_0, tlist[ind])


# calcuate and show
e_ds = []
p_ds = []
for N in N_list:
    e_d = []
    p_d = []
    for g in g_list:
        d_et = energy_dicke(g, N)
        e_d.append(d_et[0] / N)
        p_d.append(d_et[0] / d_et[1] / (N*g*np.sqrt(N)*wc**2))
        print('N={},g={}'.format(N,g))
    e_ds.append(e_d)
    p_ds.append(p_d)


# draw Fig 3(b)
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(g_list, e_ds[0], color='black', linestyle='-', label='N = 6')
plt.plot(g_list, e_ds[1], color='red', linestyle='-.', label='N = 8')
plt.plot(g_list, e_ds[2], color='blue', linestyle='--', label='N = 10')
plt.plot(g_list, e_ds[3], color='green', linestyle=':', label='N = 12')

ax = plt.gca()
ax.set_xlabel('$\\bar\\lambda$', fontsize=18, labelpad=1)
ax.set_ylabel('$E(\\tau_c)/(N\\hbar\\omega_c)$',fontsize=18,labelpad=1)
ax.set_title('Fig 3(b)', fontsize=24)
ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)

# draw Fig 3 (d)
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(g_list, p_ds[0], color='black', linestyle='-',  label = 'N = 6' )
plt.plot(g_list, p_ds[1], color='red',   linestyle='-.', label = 'N = 8' )
plt.plot(g_list, p_ds[2], color='blue',  linestyle='--', label = 'N = 10')
plt.plot(g_list, p_ds[3], color='green', linestyle=':',  label = 'N = 12')

ax = plt.gca()
ax.set_xlabel('$\\bar\\lambda$', fontsize=18, labelpad=1)
ax.set_ylabel('$P_\\bar\\lambda^{(\#)}/(Ng \\sqrt{N} \\hbar\\omega_c^2$',fontsize=18,labelpad=1)
ax.set_title('Fig 3(d)', fontsize=24)

ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)

# plt.legend(fontsize=12)


plt.show()
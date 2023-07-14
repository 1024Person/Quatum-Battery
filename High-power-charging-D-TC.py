# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			High-power-charging-D-TC
 *Date:			2023-07-10 10:06:09
 *Description:  展示Dicke模型和Tavis-Cummings模型充电的图像
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# init state
M = 15
N = 1
j = N/2.0
n = 2*j+1
hbar = 1

wa = 1
wc = 1

# init opeator
a = tensor(destroy(int(M)),qeye(int(n)))
Jm = tensor(qeye(int(M)),jmat(j,'-'))
Jp = tensor(qeye(int(M)),jmat(j,'+'))
Jz = tensor(qeye(int(M)),jmat(j,'z'))
Jx = tensor(qeye(int(M)),jmat(j,'x'))

H1 = wc*a.dag()*a + wa*Jz
H2 = 2*wc*Jx*(a+a.dag())
H3= wc*(Jm*a.dag()+Jp*a)

# init state

state_c0 = basis(M,N)
state_b0 = spin_state(j,-j)
psi_0 = tensor(state_c0,state_b0)
# calculate the average of Jz when t is equal to 0
ave_Jz_0 = expect(Jz,psi_0)
# init time set
tlist= np.linspace(0,80,10000)

# define Energ
def E1(g):
    H = H1+g*H2
    result = mesolve(H,psi_0,tlist,[],[Jz])
    Jzt = np.array(result.expect[0])
    return (Jzt - ave_Jz_0)/N
def E2(g):
    H = H1+g*H3
    result = mesolve(H,psi_0,tlist,[],[Jz])
    Jzt = np.array(result.expect[0])
    return (Jzt - ave_Jz_0) / N

# calculate and show
plt.figure(figsize=(8,6),dpi=80)
plt.plot(tlist,E1(0.05),color='blue',linestyle='-',label='Dicke model')
plt.plot(tlist,E2(0.05),color='red',linestyle='-',label='Tavis Cummings mode')

ax = plt.gca()
ax.set_xlabel('t',fontsize=18,labelpad=1)
ax.set_ylabel('max(E)/N',fontsize=18,labelpad=1)

ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)

plt.legend(fontsize=18)
plt.show()


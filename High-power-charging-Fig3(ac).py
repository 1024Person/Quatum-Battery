# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			High-power-charging-Fig2
 *Date:			2023-07-10 10:42:35
 *Description: 复现论文中的Fig2(a)
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema

# init variance
N_max = 20 # the maximal amount of the photonic
M = 40
N_list = np.linspace(1,N_max,N_max)
g_list = [0.05,0.5,2]

wb = 1
wc = 1
hbar = 1

N_t_s = 1000 # the amount of strength coupling time
N_t_w = 1000 # the amount of weak coupling time
t_s = 16 # strength coupling
t_w = 80 # weak coupling
# define the set of t
t_s_list = np.linspace(0,t_s,N_t_s)
t_w_list = np.linspace(0,t_w,N_t_w)
# because the N is vairous, the psi_0 is deferent every time
# define the energ function
def energy_dicke(g,N):
    # init vairance
    if g<0.1:
        tlist = t_w_list
    else:
        tlist = t_s_list
    j = N/2
    n = 2*j+1
    # init operator
    a = tensor(destroy(int(M)),qeye(int(n)))
    Jm = tensor(qeye(int(M)),jmat(j,'-'))
    Jp = tensor(qeye(int(M)),jmat(j,'+'))
    Jx = tensor(qeye(int(M)),jmat(j,'x'))
    Jz = tensor(qeye(int(M)),jmat(j,'z'))

    H1 = wc*a.dag()*a+wb*Jz
    H2 = 2*wc*Jx*(a.dag()+a)
    H = H1+g*H2

    ave_Jz_0 = expect(Jz,psi_0)
    result = mesolve(H,psi_0,tlist,[],[Jz])
    ave_Jz_t = np.array(result.expect[0])
    
    ind = argrelextrema(ave_Jz_t,np.greater)[0][0]
    return (ave_Jz_t[ind] - ave_Jz_0 , tlist[ind])

def energy_para(g):
    # init vairance
    if g < 0.1:
        tlist = t_w_list
    else:
        tlist = t_s_list    
    j = 1/2
    n = 2*j+1
    # init operator
    a = tensor(destroy(int(M)),qeye(int(n)))
    Jm = tensor(qeye(int(M)),jmat(j,'-'))
    Jp = tensor(qeye(int(M)),jmat(j,'+'))
    Jx = tensor(qeye(int(M)),jmat(j,'x'))
    Jz = tensor(qeye(int(M)),jmat(j,'z'))
    H1 = wc*a.dag()*a+wb*Jz
    H2 = 2*wc*Jx*(a.dag()+a)
    H = H1+g*H2

    # init state
    state_c0 = basis(int(M),int(1))
    state_b0 = spin_state(1/2,-1/2)
    psi_0 = tensor(state_c0,state_b0)
    
    # calculate the energy of QBs 
    ave_Jz_0 = expect(Jz,psi_0)
    result = mesolve(H,psi_0,tlist,[],[Jz])
    ave_Jz_t = np.array(result.expect[0])

    # find the frist maximum of energy and the time of it
    ind = argrelextrema(ave_Jz_t,np.greater)[0][0]
    return (ave_Jz_t[ind] - ave_Jz_0 , tlist[ind])



def energy_tc(g,N):
    # init vairance

    if g<0.1:
        tlist = t_w_list
    else:
        tlist = t_s_list

    j = N/2
    n = 2*j+1
    # init operator
    a = tensor(destroy(int(M)),qeye(int(n)))
    Jm = tensor(qeye(int(M)),jmat(j,'-'))
    Jp = tensor(qeye(int(M)),jmat(j,'+'))
    Jx = tensor(qeye(int(M)),jmat(j,'x'))
    Jz = tensor(qeye(int(M)),jmat(j,'z'))

    H1 = wc*a.dag()*a+wb*Jz
    H2 = wc*(Jm*a.dag()+Jp*a)
    H = H1+g*H2
    ave_Jz_0 = expect(Jz,psi_0)
    result = mesolve(H,psi_0,tlist,[],[Jz])
    ave_Jz_t = np.array(result.expect[0])

    ind = argrelextrema(ave_Jz_t,np.greater)[0][0]
    return ((ave_Jz_t[ind] - ave_Jz_0),t_w_list[ind])
   

# calculate and show
e_ds=[]
e_tcs = []
e_ps = []

p_ds = []
p_tcs = []
p_ps = []
for j in g_list:
    flag = True
    if j >0.05:
        flag =False
    e_d = []
    p_d = []
    e_tc = []
    p_tc = []
    e_ps.append(energy_para(j)[0]*np.ones(N_max))
    p_p  = []
    for i in N_list:
        # init state
        state_c0 = basis(int(M),int(i))
        state_b0 = spin_state(i/2,-i/2)
        psi_0 = tensor(state_c0,state_b0)
        d_et = energy_dicke(j,i)
        e_d.append(d_et[0]/i)
        p_d.append(d_et[0]/d_et[1]/(j*i*np.sqrt(i)*wc**2))
        p_et = energy_para(j)
        p_p.append(p_et[0]/p_et[1]/(j*i*np.sqrt(i)*wc**2))
        if flag:
            temp = energy_tc(j,i)
            e_tc.append(temp[0]/i)
            p_tc.append(temp[0]/temp[1]/(j*i*np.sqrt(i)*wc**2))
        print('i={},j={}'.format(i,j))
    e_ds.append(e_d)
    p_ds.append(p_d)
    p_ps.append(p_p)
    if flag:
        e_tcs.append(e_tc)
        p_tcs.append(p_tc)

# draw fig 3 (a)
plt.figure(figsize=(8,6),dpi = 80)
plt.scatter(N_list,e_tcs[0],color='black',marker='s',label='TC mode g = 0.05')
plt.scatter(N_list,e_ds[0],color='red',marker='o',label='Dicke mode g = 0.05')
plt.scatter(N_list,e_ds[1],color='blue',marker='^',label='Dicke mode g = 0.5')
plt.scatter(N_list,e_ds[2],color='green',marker='D',label='Dicke mode g = 2')
plt.plot(N_list,e_ps[0],color='red',linestyle='-.')
plt.plot(N_list,e_ps[1],color='blue',linestyle='dotted')
plt.plot(N_list,e_ps[2],color='green',linestyle='--')
ax = plt.gca()
ax.set_xlabel('N',fontsize=18,labelpad=1)
ax.set_ylabel('$E(\\tau_c)/(N\\hbar\\omega_c)$',fontsize=18,labelpad=1)
ax.set_title('Fig3 (a)',fontsize=24)
ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)


# draw Fig3 (c)
plt.figure(figsize=(8,6),dpi = 80)
plt.scatter(N_list,p_tcs[0],color='black',marker='s',label='Dicke mode g = 0.05')
plt.scatter(N_list,p_ds[0],color='red',marker='o',label='Dicke mode g = 0.05')
plt.scatter(N_list,p_ds[1],color='blue',marker='^',label='Dicke mode g = 0.5')
plt.scatter(N_list,p_ds[2],color='green',marker='D',label='Dicke mode g = 2')

plt.plot(N_list,p_ps[0],color='red',linestyle='-',label='Dicke mode g = 0.05')
plt.plot(N_list,p_ps[1],color='blue',linestyle='-.',label='Dicke mode g = 0.5')
plt.plot(N_list,p_ps[2],color='green',linestyle='--',label='Dicke mode g = 2')
ax = plt.gca()
ax.set_xlabel('N',fontsize=18,labelpad=1)
ax.set_ylabel('$P_\\bar\\lambda^{(\#)}/(Ng \\sqrt{N} \\hbar\\omega_c^2$',fontsize=18,labelpad=1)
ax.set_title('Fig3(c)',fontsize=24)
ax.axis([1,20,0,1])
plt.show()


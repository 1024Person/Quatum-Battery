"""
论文的Fig1，算法重建
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math

# init variance
M = 15 # the dimension of a
N = 10 # the amount of photonic
j = N/2.0 # the enegevalues of J^2
m = 2*j+1 # the enegevalues of Jz

wa = 1
wc = 1

# init opeators

# ============================== this part is ambiguous for me.===============
a = tensor(destroy(int(M)),qeye(int(m))) 
Jm = tensor(qeye(int(M)),jmat(int(j),'-'))
Jp = tensor(qeye(int(M)),jmat(int(j),'+'))
Jz = tensor(qeye(int(M)),jmat(int(j),'z'))
Jx = tensor(qeye(int(M)),jmat(int(j),'x'))

H1 = wc*a.dag()*a + wa*Jz
H2 = 2*wc*Jx*(a+a.dag())

# init states
psi_b0 = spin_state(j,-j)
psi_c0 = basis(M,N) # what does mean?
psi_0 = tensor(psi_c0,psi_b0)

# t set
tlist = np.linspace(0,100,1000)

# calcuate average of H(0) and H(τc)
J_ave_0 = expect(Jz,psi_0)

# define function named E,and input parameter g to which lambda limit 
def E(g,t):
    H = H1+g*H2
    result = mesolve(H,psi_0,t,[],[Jz])
    Jzt = np.array(result.expect[0])

    return (Jzt - J_ave_0)/N
# draw the dynamic change of E(t)
# yeah, this selection is copied form zhihu.com since it is little important.

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(111)
ax=plt.gca()
plot1,=plt.plot(tlist,E(g=0.5,t=tlist),color="green",linewidth=2.5,label='g = 0.5',linestyle="-.")
plot2,=plt.plot(tlist,E(g=2.0,t=tlist),color="blue",linewidth=2.5,label='g = 2.0',linestyle="-")


plt.xticks(fontsize=22)#对坐标的值数值，大小限制
plt.yticks(fontsize=22)

ax=plt.gca() 
plt.axis([0,8,0,1])

ax.set_ylabel('max(E)/N',fontsize=22,labelpad = 1)
ax.set_xlabel('t',fontsize=22,labelpad =1)

ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)

plt.legend(fontsize=18)
# plt.show()
# g << 1
tlist_ = np.linspace(0,80,5000)
plt.figure(figsize=(8,6),dpi=80)
plot3, = plt.plot(tlist_,E(0.05,tlist_),'r-',label='g << 1',color='red',linestyle='-')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax = plt.gca()
plt.axis([0,80,0,1])
ax.set_xlabel('t',fontsize=22,labelpad=1)
ax.set_ylabel('max(E)/N',fontsize=22,labelpad=1)

ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
plt.legend(fontsize='18')
plt.show()
# 




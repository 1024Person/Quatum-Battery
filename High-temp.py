from qutip import *
from pylab import *
import numpy as np
import math
import matplotlib.pyplot as plt

M = 15#dimesion of a
N = 10#atoms 
N1=600#times 
N2=10#photons

j = N/2.0
n = 2*j + 1

wc = 1.0
w0 = 1.0


a  = tensor(destroy(int(M)), qeye(int(n)))
Jp = tensor(qeye(int(M)), jmat(int(j), '+'))
Jm = tensor(qeye(int(M)), jmat(int(j), '-'))
Jz = tensor(qeye(int(M)), jmat(int(j), 'z'))
Jx = tensor(qeye(int(M)), jmat(int(j), 'x'))

H0 = wc * a.dag() * a + w0 * Jz
H1 =2*wc*Jx * (a + a.dag())

#initial state
psi_a0 = spin_state(j, -j)
psi_c0=basis(M,N2)
psi0=tensor(psi_c0,psi_a0)

#
aver_Jz0=expect(Jz,psi0)

#times
t=np.linspace(0,8,1000)

def E(g):

    H=H0+g*H1
    result=mesolve(H,psi0,t,[],[Jz])
    Jzt = np.array(result.expect[0])
    return (Jzt-aver_Jz0)/N

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(111)  
ax=plt.gca()
plot1,=plt.plot(t,E(g=0.5),color="green",linewidth=2.5,label='g=0.5',linestyle="-.")
plot2,=plt.plot(t,E(g=2.0),color="blue",linewidth=2.5,label='g=2.0',linestyle="-")

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
plt.show()
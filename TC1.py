import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math

N = 8      # atoms number
M = 4*N # dimesion of a

j = N/2.0

wc = 1.0
w0 = 1.0


a = tensor(destroy(M), qeye(N+1))
Jp = tensor(qeye(M), jmat(j, '+'))
Jm = tensor(qeye(M), jmat(j, '-'))
Jz = tensor(qeye(M), jmat(j, 'z'))
Jx = tensor(qeye(M), jmat(j, 'x'))
jz = jmat(j, 'z')

# jp = jmat(j, '+')
# jm = jmat(j, '-')

H0 = wc * a.dag() * a
H1 = w0 * (Jz+N/2)
Hqb1 = w0 * (jz+N/2)

# H2 = wc*(Jp+Jm)*(a.dag()+a)
H2 = wc*(Jp*a+Jm*a.dag())
g = 1
H = H0+H1+g*H2

vb = np.array(sorted(Hqb1.eigenenergies()))
vb2 = vb.reshape(len(vb), 1)
# initial state
psi_a0 = spin_state(j, -j)

# fock态
psi_c0 = basis(M, N)
psi0 = tensor(psi_c0, psi_a0)

# 相干态
psi_c1 = coherent(M, np.sqrt(N))
psi1 = tensor(psi_c1, psi_a0)

# 压缩态
vac = basis(M, 0)
s = squeeze(M, math.asinh(np.sqrt(N)))
psi_c2 = s*vac
psi2 = tensor(psi_c2, psi_a0)


tlist = np.linspace(0, 2*np.pi, 699)


def e0(tlist):
    eb = []     # 可提取功
    et1 = []    # 能量
    # e11 = []  # 比值
    M = []
    result = mesolve(H, psi1, tlist, [], [])
    psit = result.states
    for i in range(len(psit)):
        psi0t = psit[i]
        rhob = ptrace(psi0t, 1)
        et = np.trace(Hqb1 * rhob)
        et1.append(abs(et)/N)

        vb1 = np.array(sorted(rhob.eigenenergies(), reverse=True))
        # vb2 = vb1.reshape(len(vb1), 1)
        m = np.dot(vb1, vb2)
        M.append(m)
        e = abs(et - m)
        eb.append(e/N)
        # e1 = e/et
        # e11.append(e1)

    return eb, et1, M
# e11,
print(e0(tlist)[2])

aver_H10 = expect(H1, psi0)
print(aver_H10)
def E0(tlist):
    result = mesolve(H, psi1, tlist, [], [H1])
    H0t = np.array(result.expect[0])
    return (H0t-aver_H10)/N
aver_H11=expect(H1,psi1)

# def E1(tlist):
#     result=mesolve(H,psi1,tlist,[],[H1])
#     H1t = np.array(result.expect[0])
#     return (H1t-aver_H11)/N


# aver_H01=expect(H0,psi1)
#
#
# def E10(t):
#
#     result=mesolve(H,psi1,t,[],[H0])
#     H10t = np.array(result.expect[0])
#     return (H10t-aver_H01)/N
#
#
# aver_H12=expect(H1,psi2)


# def E2(t):
#
#     result=mesolve(H,psi2,t,[],[H1])
#     H2t = np.array(result.expect[0])
#     return (H2t-aver_H12)/N


plt.figure(figsize=(8, 6), dpi=80)
plt.plot(111)
ax=plt.gca()
plot1,=plt.plot(np.sqrt(N)*g*tlist,E0(tlist),color="black",linewidth=2.5,label='coherent state',linestyle="-")
plot2,=plt.plot(np.sqrt(N)*g*tlist,e0(tlist)[0],color="red",linewidth=2.5,linestyle="--")
plot2,=plt.plot(np.sqrt(N)*g*tlist,e0(tlist)[1],color="green",linewidth=2.5,linestyle="-")
# plot3,=plt.plot(np.sqrt(N)*g*tlist,e0(tlist)[2],color="green",linewidth=2.5,linestyle="--")
# plot3,=plt.plot(np.sqrt(N)*g*tlist,E1(tlist),color="black",linewidth=2.5,label='coherent state',linestyle="-")
# plot1,=plt.plot(np.sqrt(N)*g*t,E10(t),color="blue",linewidth=2.5,label='coherent state',linestyle="-")
# plot3,=plt.plot(np.sqrt(N)*g*t,E2(t),color="green",linewidth=2.5,label='squeeze state',linestyle="--")
plt.xticks(fontsize=16) # 对坐标的值数值，大小限制
plt.yticks(fontsize=16)


ax=plt.gca()
plt.axis([0,2*np.pi,-0.05,1])

# ax.set_ylabel('E/N$\omega$0',fontsize=16,labelpad = 1)
ax.set_xlabel('$\sqrt{N}$gt',fontsize=16,labelpad =1)

ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)

plt.legend(fontsize=18)
plt.show()

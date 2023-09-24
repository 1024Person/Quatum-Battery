import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import time
import math
import random
from tqdm import tqdm
from scipy.sparse.linalg import expm


g = basis(3, 0)  # |g>
e = basis(3, 1)  # |e>
f = basis(3, 2)  # |f>

s_eg = e * g.dag()  # |e><g|
s_fe = f * e.dag()  # |f><e|
s_ge = s_eg.dag()   # |g><e|
s_ef = s_fe.dag()   # |e><f|
s_ff = f * f.dag()  # |f><f|
s_gg = g * g.dag()  # |g><g|
s_ee = e * e.dag()  # |e><e|


def Is1(i): return [qeye(3) for j in range(i)]
def basis1(i): return [g for j in range(i)]


def ad(Nc, N): return tensor([destroy(Nc)] + Is1(N))
def a_(Nc, N): return tensor([destroy(Nc).dag()] + Is1(N))


def S_ff(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_ff] + Is1(N - i-1))
def S_ee(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_ee] + Is1(N - i-1))
def S_gg(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_gg] + Is1(N - i-1))
def S_eg(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_eg] + Is1(N - i-1))
def S_ge(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_ge] + Is1(N - i-1))
def S_fe(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_fe] + Is1(N - i-1))
def S_ef(Nc, N, i): return tensor([qeye(Nc)] + Is1(i) + [s_ef] + Is1(N - i-1))


def S_ff1(N, i): return tensor(Is1(i) + [s_ff] + Is1(N - i-1))
def S_ee1(N, i): return tensor(Is1(i) + [s_ee] + Is1(N - i-1))
def S_gg1(N, i): return tensor(Is1(i) + [s_gg] + Is1(N - i-1))


def psi1(Nc, N): return tensor([fock(Nc, M)] + basis1(N))
def psi2(Nc, N): return tensor([coherent(Nc, np.sqrt(M))] + basis1(N))


N = 4  # 三能级原子数
M = 2*N
Nc = 8*N+1 # 希尔伯特空间截断数

tlist = np.linspace(0.0001, 30, 999)
vac=basis(Nc,0)
s=squeeze(Nc,math.asinh(np.sqrt(M)))
psi_c0=s*vac

def psi3(Nc, N): return tensor([psi_c0] + basis1(N))


def Ham0(N):
    wg = 0
    we = 1
    wf = 1.95
    H0 = 0
    for i in range(N):
        H0 += wf * S_ff1(N, i) + we * S_ee1(N, i) + wg * S_gg1(N, i)
    return H0


vb = np.array(Ham0(N).eigenenergies())


def Ham(Nc,N):
    wc = 1  # 腔频率
    wg = 0
    we = 1
    wf = 1.95

    g1 = 1   # 耦合强度
    g2 = 1

    H0 = 0
    H1 = 0

    Hc = wc * a_(Nc, N) * ad(Nc, N)
    for i in range(N):
        # H01 += wf * S_ff1(Nc, N, i) + we * S_ee1(Nc, N, i) + wg * S_gg1(Nc, N, i)
        H0 += wf * S_ff(Nc, N, i) + we * S_ee(Nc, N, i) + wg * S_gg(Nc, N, i)
        H1 += g1/np.sqrt(N) * (a_(Nc, N) + ad(Nc, N)) * (S_eg(Nc, N, i) + S_ge(Nc, N, i))\
              + g2/np.sqrt(N) * (a_(Nc, N) + ad(Nc, N)) * (S_fe(Nc, N, i) + S_ef(Nc, N, i))

    H = Hc + H0 + H1

    return H, H0


def e0(Nc,N):
    H = Ham(Nc, N)[0]
    H0 = Ham0(N)
    psi0 = psi2(Nc, N)
    eb = []
    ET=[]
    result0 = mesolve(H, psi0, tlist, [], [])
    psit0 = result0.states
    for i in range(len(psit0)):

        # psi = np.array(psit0[i])
        # psit = psi.reshape([Nc,3**N])
        # rho1 = np.dot(psit.conjugate().transpose(), psit)  # 约化密度矩阵

        rho1 = ptrace(psit0[i],[1,2,3,4])

        Et = np.trace(H0 * rho1)
        ET.append(Et)

        # evals, evecs = np.linalg.eigh(rho1)
        evals = np.array(rho1.eigenenergies())
        vb11 = np.array(sorted(evals, reverse=True))
        vb22 = vb11.reshape(len(vb11), 1)

        m = np.dot(vb, vb22)
        e = abs(Et-m)
        eb.append(e)
    return ET
# print(e0(Nc,N))

# np.savetxt('E:/3/E.txt', e0(Nc,N)[0], fmt="%.4e", delimiter=" ")
# np.savetxt('E:/3/e.txt.txt', e0(Nc,N)[1], fmt="%.4e", delimiter=" ")
fig = plt.figure(figsize = (8,6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.plot(tlist, e0(Nc,N)[0],color="black", linestyle="-", linewidth=2.3)
plt.plot(tlist, e0(Nc,N),color="black", linestyle="-", linewidth=2.3)
# plt.plot(tlist, e0(Nc,N)[1],color="red", linestyle="-", linewidth=2.3)
plt.tick_params(labelsize=18,top='on', right='on', which='both',length=3.5, width=2.5,pad=5, )
plt.xlabel('$gt$', fontsize=18, labelpad=1)  # 横坐标标注尺寸
plt.axis([0, 30, 0, 8.01])  # 范围
my_x_ticks = np.arange(0, 8.00001, 4)
my_y_ticks = np.arange(0, 8.00001, 4)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
# plt.legend(fontsize=18,frameon=False)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)

plt.show()



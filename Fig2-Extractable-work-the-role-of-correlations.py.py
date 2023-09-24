from qutip import *
import numpy as np
import matplotlib.pyplot as plt

N = 8
tao = 2 * np.pi / np.sqrt(N)
tn = 500
t = np.linspace(0, tao, tn)
M = 4 * N
g = 1
hbar = 1
omega0 = 1
j = N / 2
n = 2 * j + 1
alpha = np.sqrt(N)
sub_sigma = []
plu_sigma = []
sz_sigma = []
# 单位矩阵
si = qeye(2)
# 构造湮灭算符和产生算符
a = tensor(si, si, si, si, si, si, si, si, destroy(M))
a_dag = a.dag()
# 构造单粒子的升降算符,在循环中会用到
sup = jmat(1 / 2, '+')
sdown = jmat(1 / 2, '-')
sz = jmat(1 / 2, 'z')
# 构造Fock态
psi_0 = []
# 创建含有8个单位矩阵的列表
op_list = []
for j in range(N):
    op_list.append(si)
# 构造集体算符，σ正和σ负
for i in range(N):
    # 构造sigma+算符
    op_list[i] = sup
    temp = tensor(op_list)
    plu_sigma.append(temp)
    # 构造sigma-算符
    op_list[i] = sdown
    temp = tensor(op_list)
    sub_sigma.append(temp)
    # 构造sigmaz算符
    op_list[i] = sz
    temp = tensor(op_list)
    sz_sigma.append(temp)
    # 重新把第i个元素赋值为单位矩阵，因为下一次循环还需要用到这个列表
    # 如果，不现在把这个元素改回来，下次可能会出错
    op_list[i] = si
    # 当前粒子的fock态
    psi_0.append(basis(2, 1))
psi_0 = tensor(psi_0)
# 构造体系初态
psi_coh = coherent(M, alpha)  # 相干态
psi_fork = basis(M,0)
psi_squ = squeeze(M,np.arcsinh(N))*basis(M,0)
# 富克态和相干态的直积
psi_all = tensor(psi_0, psi_coh)


# 计算量子电池的能量，可提取功，功率以及充电器的能量
def WE_calc(g):
    """
    计算量子电池的能量，EB
    参数g代表耦合强度
    """
    # 构造充电器的哈密顿量
    HA = omega0 * a_dag * a
    # 构造电池的哈密顿量和相互作用哈密顿量
    HB = 0
    H1 = 0
    HB1 = 0
    for i in range(N):
        HB += omega0 * tensor(plu_sigma[i], qeye(M)) * tensor(sub_sigma[i], qeye(M))
        H1 += g * (a * tensor(plu_sigma[i], qeye(M)) + a_dag * tensor(sub_sigma[i], qeye(M)))
        HB1 += omega0 * plu_sigma[i] * sub_sigma[i]

    # 体系哈密顿量
    H = HA + HB + H1
    EB0 = expect(HB, psi_all)
    result = mesolve(H, psi_all, t, [], [])
    states = result.states
    E_B = np.ones_like(t)
    W = np.ones_like(t)
    E_A = np.ones_like(t)
    vb2 = np.array(HB1.eigenenergies())
    for i in range(tn):
        psi = states[i]
        rhoB = ptrace(psi, [_ for _ in range(N)])
        E_B[i] = expect(HB, psi)
        E_A[i] = expect(HA, psi)
        vb1 = np.array(rhoB.eigenenergies(sort='high'))  # 降序排列
        m = np.sum(vb1 * vb2)
        W[i] = E_B[i] - m
        E_B[i] = E_B[i] - EB0
        print('i={}'.format(i))

    return E_B, W, W / (E_B+0.0001), E_A


EB, WB, WE_r, EA = WE_calc(g=1)
plt.plot(np.sqrt(N) * t, EB / N, color='black', linestyle='-', label=r'$E_B^{(N)}(\tau)$')
plt.plot(np.sqrt(N) * t, WB / N, color='red', linestyle='--', label=r'$\epsilon^{(N)}(\tau)$')
plt.plot(np.sqrt(N) * t, EA / N, color='blue', linestyle=':', label=r'$E_A^{(N)}(\tau)$')
plt.plot(np.sqrt(N) * t, WE_r, color='green', linestyle='-.', label=r'$\frac{\epsilon^{(N)}_B(\tau)}{E_B^{(N)}(\tau)}$')
plt.xlim(0, 2 * np.pi)
plt.ylim(0, 1.1)
plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
plt.xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
           [r'0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
plt.legend()
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.show()

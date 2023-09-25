from qutip import *
import numpy as np
from tools.extractable import passive_e
import matplotlib.pyplot as plt


def calc_ratio(n):
    print('开始计算N={}'.format(n))
    # 初始化变量
    g = 1
    t_total = 2 * np.pi
    tn = 500
    tlist = np.linspace(0.00001, t_total, tn)
    alpha = np.sqrt(n)
    sq = np.arcsinh(n)
    M = 4 * n
    omega0 = 1
    s_up = jmat(1 / 2, '+')
    s_down = jmat(1 / 2, '-')
    s_up_list = []
    s_down_list = []
    op_list = []
    psi_f = []
    si = qeye(2)
    # 构造共同希尔伯特空间的直积算符
    op_list = [si for _ in range(n)]
    # for i in range(n):
        # op_list.append(si)
    for i in range(n):
        # 构造σ+算符
        op_list[i] = s_up
        s_up_list.append(tensor(op_list))
        # 构造σ-算符
        op_list[i] = s_down
        s_down_list.append(tensor(op_list))
        # 构造fork态
        psi_f.append(basis(2, 1))
        # 重置一下op_list这个列表，下一次循环还需要用到
        op_list[i] = si
    # 直积构造量子电池fork态
    psi_f = tensor(psi_f)
    # 构造产生湮灭算符
    a = tensor(tensor(op_list), destroy(M))
    a_dag = a.dag()
    # 构造哈密顿量
    HA = omega0 * a_dag * a
    HB = 0
    HI = 0
    HB1 = 0
    for i in range(n):
        HB += omega0 * tensor(s_up_list[i], qeye(M)) * tensor(s_down_list[i], qeye(M))
        HI += g * (a * tensor(s_up_list[i], qeye(M)) + a_dag * tensor(s_down_list[i], qeye(M)))
        HB1 += omega0 * s_up_list[i] * s_down_list[i]
    H = HA + HB + HI
    # 开始构造三个初态
    # 腔的态，腔中的能量刚开始要设置生N*omega0和电池的容量相匹配
    psi_coh = coherent(M, alpha=alpha)
    psi_fork = basis(M, n)
    vac = basis(M, 0)
    psi_squ = squeeze(M, sq) * vac
    # 整个体系的态
    psi_01 = tensor(psi_f, psi_fork)
    psi_02 = tensor(psi_f, psi_coh)
    psi_03 = tensor(psi_f, psi_squ)
    # 开始进行态的演化
    result1 = mesolve(H, psi_01, tlist, [], [])
    result2 = mesolve(H, psi_02, tlist, [], [])
    result3 = mesolve(H, psi_03, tlist, [], [])
    # 寻找最优时间
    e = expect(HB,result1.states)
    p = e/tlist
    ind_tau1 = np.where(p == np.max(p))[0][0]

    e = expect(HB,result2.states)
    p = e/tlist
    ind_tau2 = np.where(p == np.max(p))[0][0]
    
    e = expect(HB,result3.states)
    p = e/tlist
    ind_tau3 = np.where(p == np.max(p))[0][0]
    
    # 准备需要的态
    psi_1 = result1.states[ind_tau1]
    psi_1B = ptrace(psi_1, [_ for _ in range(n)])
    psi_2 = result2.states[ind_tau2]
    psi_2B = ptrace(psi_2, [_ for _ in range(n)])
    psi_3 = result3.states[ind_tau3]
    psi_3B = ptrace(psi_3, [_ for _ in range(n)])
    # 计算能量
    E1 = expect(HB, psi_1)  # fork态的能量
    m1 = passive_e(HB1,psi_1B) # fork态的被动态能量
    W1 = E1 - m1
    E2 = expect(HB, psi_2)  # 相干态的电池能量
    m2 = passive_e(HB1,psi_2B) # 相干态的被动态能量
    W2 = E2 - m2
    E3 = expect(HB, psi_3)  # 压缩态的电池能量
    m3 = passive_e(HB1,psi_3B) # 压缩态的被动态能量
    W3 = E3 - m3
    return W1 / (E1 + 0.0001), W2 / (E2 + 0.0001), W3 / (E3 + 0.0001)


N = np.array([1, 2, 3, 4, 5, 6, 7, 8])
we1 = np.ones_like(N, dtype=float)
we2 = np.ones_like(N, dtype=float)
we3 = np.ones_like(N, dtype=float)
for i in range(len(N)):
    we1[i], we2[i], we3[i], = calc_ratio(N[i])

print('计算完毕，开始画图')
plt.figure()
plt.scatter(N, we3, marker='s', color='green', label='squeeze')
plt.scatter(N, we1, marker='o', color='red', label='fock')
plt.scatter(N, we2, marker='^', color='blue', label='coherent')
plt.ylabel(r'$\epsilon_B^{(N)}(\bar{\tau})/E^{(N)}_B(\bar{\tau})$')
plt.xlabel(r'$N$')
# plt.xlim(0.5, 8.5)
# plt.xticks([2, 4, 6, 8], ['2', '4', '6', '8'])
# plt.ylim(-0.1, 1.1)
# plt.yticks([0, 0.5, 1], ['0', r'$\frac{1}{2}$', '1'])
plt.legend()
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.show()

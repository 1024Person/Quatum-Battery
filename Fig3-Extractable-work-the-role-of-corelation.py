"""
通过fig2的代码发现:fig2的代码跑出来的富克态是正确的，
所以重新开一个file,先把富克态的写对再说
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt


omega0 = 1
g = 1
t_total = 2*np.pi
tn = 500
t = np.linspace(0.000001,t_total,tn)
def calc_ration(n):
    """
    计算量子电池可提取功和能量的比例
    """
    print('开始计算n={}'.format(n))
    M = 4*n
    alpha = np.sqrt(n)
    sub_sigma = []
    plu_sigma = []
    # 单位矩阵
    si = qeye(2)
    # 构造产生湮灭算符
    a = tensor(tensor([si for _ in range(n)]), destroy(M))
    a_dag = a.dag()
    # 构造单粒子的升降算符
    sup = jmat(1/2,'+')
    sdown = jmat(1/2,'-')
    # 构造Fock态
    psi_0 = []
    # 创建含有n个单位矩阵的列表
    op_list = [si for _ in range(n)]
    # 构造集体希尔伯特空间中的算符，σ正和σ负    
    for i in range(n):
        # 构造σ+算符
        op_list[i] = sup
        temp = tensor(op_list)
        plu_sigma.append(temp)
        # 构造σ-算符
        op_list[i] = sdown
        temp = tensor(op_list)
        sub_sigma.append(temp)
        # 重新把第i个元素赋值为单位矩阵，因为下一次循环还需要用到op_list
        # 如果不现在把这个元素改回来，下次就不是和单位矩阵直积了 
    psi_0 = tensor([basis(2,1) for _ in range(n)])
    # 构造充电器的fork态，注意充电器中刚开始是有能量的，
    # 并且能量和量子电池的容量是相互匹配的
    psi_fork = basis(M,n)
    # 构造体系初态
    psi_all = tensor(psi_0,psi_fork)

    # 计算量子电池的能量，可提取功，功率，以及充电器的能量
    # 构造哈密顿量
    HA = omega0*a_dag*a # 充电器的哈密顿量
    HB = Qobj() # 电池的哈密顿量
    HI = Qobj() # 电池和充电器相互作用的哈密顿量
    HB1 = Qobj() # 在电池空间中，电池的哈密顿量，这个在之后计算被动态的时候，有用
    for i in range(n):
        HB += omega0 * tensor(plu_sigma[i],qeye(M))*tensor(sub_sigma[i],qeye(M))
        HI += g*(a*tensor(plu_sigma[i],qeye(M))+a_dag*tensor(sub_sigma[i],qeye(M)))
        HB1+= omega0*plu_sigma[i]*sub_sigma[i]

    # 体系哈密顿量
    H = HA+HB+HI
    EB0 = expect(HB,psi_all)
    result = mesolve(H,psi_all,t,[],[])
    states = result.states
    E_B = np.ones_like(t)
    W = np.ones_like(t)
    E_A = np.ones_like(t)
    vb2 = np.array(HB1.eigenenergies())
    for i in range(tn):
        psi = states[i]
        rhoB = ptrace(psi, [_ for _ in range(n)])
        E_B[i] = expect(HB,psi)
        E_A[i] = expect(HA,psi)
        vb1 = np.array(rhoB.eigenenergies(sort='high')) # 降序排列
        m = np.sum(vb1*vb2)
        W[i] = E_B[i] - m
        E_B[i] = E_B[i] - EB0
        # print('i={}'.format(i))
    
    P = E_B / t
    ind_tau = np.where(P == np.max(P))
    
    return W[ind_tau] / E_B[ind_tau]

N = [1,2,3,4,5,6,7,8]
ratio_fork = np.ones_like(N)
for i in range(len(N)):
    ratio_fork[i] = calc_ration(N[i])
plt.scatter(N,ratio_fork)
plt.show()


from qutip import *
import numpy as np
import matplotlib.pyplot as plt

tao =2*np.pi
tn = 1000
t = np.linspace(0,tao,tn)
N = 8
M = 4*N
g = 1
hbar = 1
omega0 = 1
j = N/2
n = 2*j+1
alpha = np.sqrt(N)
sub_sigma = []
plu_sigma = []
sz_sigma = []
# 单位矩阵
si = qeye(2)
# 构造湮灭算符和产生算符
a = tensor(si,si,si,si,si,si,si,si,destroy(M))
a_dag = a.dag()
# 构造单粒子的升降算符,在循环中会用到
sup = jmat(1/2,'+')
sdown = jmat(1/2,'-')
sz = jmat(1/2,'z')
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
    psi_0.append(basis(2,0))
psi_0 = tensor(psi_0)
# 构造体系初态
s_al = coherent(M,alpha) # 相干态
# 富克态和相干态的直积
psi_all = tensor(psi_0,s_al)

# 计算量子电池的能量，可提取功，功率以及充电器的能量
def WE_calc(g):
    """
    计算量子电池的能量，EB
    参数g代表耦合强度
    """
    # 构造充电器的哈密顿量
    HA = omega0*a_dag*a
    # 构造电池的哈密顿量和相互作用哈密顿量
    HB = 0
    H1 = 0
    HB1 = 0
    for i in range(N):
        HB+=omega0*tensor(sz_sigma[i],qeye(M))
        H1+=g*(a*tensor(plu_sigma[i],qeye(M))+a_dag*tensor(sub_sigma[i],qeye(M)))
        HB1 += omega0*sz_sigma[i]

    # 体系哈密顿量
    H = HA+HB+H1
    EB0 = expect(HB,psi_all)
    result = mesolve(H,psi_all,t,[],[])
    states = result.states
    E = np.ones_like(t)
    W = np.ones_like(t)
    for i in range(tn):
        psi = states[i]
        rhoB = ptrace(psi,[0,1,2,3,4,5,6,7])
        E[i] = np.trace(HB1*rhoB)
        vb1 = np.array(sorted(rhoB.eigenenergies(),reverse=True)) # 降序排列
        vb2 = HB1.eigenenergies()
        m = np.sum(vb1*vb2)
        W[i] = E[i] - m
        E[i] = E[i] - EB0
        print('i={}'.format(i))

    return np.abs(E),np.abs(W)

E1,W1 = WE_calc(g=1)
plt.plot(t*np.sqrt(N)*g,E1/N,color='black',linestyle='-')
plt.plot(t*np.sqrt(N)*g,W1/N,color='red',linestyle='--')

plt.show()


# for i in range(gn):
    # EB[i]=E_calc(g=gs[i])
    # print('{},gs={}'.format(i,gs[i]))
# plt.plot(np.sqrt(N)*gs*tao,EB)

# plt.show()

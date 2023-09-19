import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import time
import math
import random
from tqdm import tqdm
from scipy.sparse.linalg import expm
"""
时序图
"""

# start_time = time.time()

# def random_int_list(start, stop, length):  # random list
#     start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
#     length = int(abs(length)) if length else 0
#     random_list = []
#     for i in range(length):
#         random_list.append(0.01*random.randint(start, stop))
#     return random_list

# def psi(N, J, omega_a, delta):
#     sup = jmat(1 / 2, '+')
#     sdown = jmat(1 / 2, '-')
#     si = qeye(2)  # Unit matrix
#     sup_list = []
#     sdown_list = []
#
#     for n in range(N):
#         op_list = []
#         for m in range(N):
#             op_list.append(si)
#
#         op_list[n] = sup
#         sup_list.append(tensor(op_list))
#
#         op_list[n] = sdown
#         sdown_list.append(tensor(op_list))
#
#     HB1 = 0
#     for n in range(N):
#         HB1 += omega_a * sup_list[n] * sdown_list[n]
#
#     for n in range(N - 1):
#         if n in [0, 2, 4, 6, 8]:
#             HB1 += -J[n] * (1 + delta) * (sup_list[n] * sdown_list[n + 1] + sup_list[n+1] * sdown_list[n])
#
#     for n in range(N - 1):
#         if n in [1, 3, 5, 7]:
#             HB1 += -J[n] * (1 - delta) * (sup_list[n] * sdown_list[n + 1] + sup_list[n+1] * sdown_list[n])
#
#     psi_all = (HB1).groundstate()[1]
#     return psi_all


def E_fun_001(N, J, omega_a, omega_c, delta, g, tlist):
    nc = 10 * N
    si = qeye(2)  # Unit matrix单位阵
    a = tensor(si, si, si,  destroy(nc + 1))

    sup = jmat(1 / 2, '+')
    sdown = jmat(1 / 2, '-')
    sz = jmat(1 / 2, 'z')

    sup_list = []
    sdown_list = []
    sz_list = []
    mat_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sup
        sup_list.append(tensor(op_list))

        op_list[n] = sdown
        sdown_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

        op_list[n] = si
        mat_list.append(tensor(op_list))

    # construct the hamiltonian
    HS = 0
    HA = 0
    HB = 0
    HI = 0
    mat_1 = 0

    for n in range(N):
        # print(n)
        mat_1 += tensor(mat_list[n], qeye(nc + 1))
        # HB += omega_a * tensor(sup_list[n], qeye(nc + 1)) * tensor(sdown_list[n], qeye(nc + 1))
        HB += omega_a * tensor(sz_list[n], qeye(nc + 1))
        # HI += g * (tensor(sup_list[n],qeye(nc+1)) * a + a.dag() * tensor(sdown_list[n],qeye(nc+1)))
        HI += g * (tensor(sup_list[n], qeye(nc + 1)) + tensor(sdown_list[n], qeye(nc + 1))) * (a + a.dag())


    # for n in range(N - 1):
    #     if n in [0,2,4,6,8]:
    #         HB += -J[n] * (1+delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
    #         HB += -J[n] * (1+delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))
    #
    #
    # for n in range(N - 1):
    #     if n in [1,3,5,7]:
    #         HB += -J[n] * (1-delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
    #         HB += -J[n] * (1-delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))

    HA += omega_c * a.dag() * a


    HS = HA + HB + HI

    # mat_1 = mat_1 / N
    # Emax = max(HS.eigenenergies())
    # Emin = min(HS.eigenenergies())
    # HS = (1 / (Emax - Emin)) * (2 * HS - (Emax + Emin) * mat_1)
    # psi_1=psi(N, J, omega_a, delta)
    # psi_all = tensor(psi(N, J, omega_a, delta), coherent(nc+1, np.sqrt(0.25)))
    # np.savetxt('D:/python/python001/QB/data/HS2.txt', psi_1, fmt="%.4e", delimiter=" ")
    # print(psi_all)
    # print(psi_1)

    # print(shape(HS.eigenenergies()))
    # print(shape(psi_all))

    psi_list = []
    for n in range(N):
        psi_list.append(basis(2, 0))
    # # psi_list.append(basis(2, 1)
    psi0 = tensor(psi_list)
    # 相干态第一个参数是截断数，第二个参数是alpha，但是第二个参数为什么是2*N？难道不应该是N吗？
    # 还有就是前面的这个nc+1是什么意思？前面的这个希尔伯特空间截断数应该是随便设得，
    # 但是这里的这个α不应该是随便设的吧
    psi_all=tensor(psi0,coherent(nc+1, np.sqrt(N)))    #相干态

    # psi_all = tensor(psi0, basis(nc+1, 2*N))        # 富克态      

    # vac = basis(nc+1, 0)
    # s = squeeze(nc+1, math.asinh(np.sqrt(2*N)))            #squeeze态
    # psi_c0 = s * vac
    # psi_all = tensor(psi0, psi_c0)
    # psi_all = tensor(psi(N, J, omega_a, delta), basis(nc + 1, nc))
    E0 = expect(HB, psi_all)
    # 这个应该是能量吧？
    def E(tlist):
        result = mesolve(HS, psi_all, tlist, [], [HB])
        Et = np.array(result.expect[0])
        return abs(Et-E0)
    P=E(tlist)/tlist # 这里的P为什么是E / tlist?难道不应该先求一下差分，在搞吗？
    return E(tlist),P

# def E_fun_002(N, J, omega_a, omega_c, delta, g, tlist):
#     nc = 3 * N
#     si = qeye(2)  # Unit matrix
#     a = tensor(si, si, si, si, destroy(nc+1))
#     sup = jmat(1 / 2, '+')
#     sdown = jmat(1 / 2, '-')
#
#
#     sup_list = []
#     sdown_list = []
#     mat_list = []
#
#     for n in range(N):
#         op_list = []
#         for m in range(N):
#             op_list.append(si)
#
#         op_list[n] = sup
#         sup_list.append(tensor(op_list))
#
#         op_list[n] = sdown
#         sdown_list.append(tensor(op_list))
#
#         op_list[n] = si
#         mat_list.append(tensor(op_list))
#
#     # construct the hamiltonian
#     HS=0
#     HA=0
#     HB=0
#     HI=0
#     mat_1 = 0
#
#     for n in range(N):
#         # print(n)
#         mat_1 += tensor(mat_list[n],qeye(nc+1))
#         HB += omega_a * tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1))
#         HI += g * (tensor(sup_list[n],qeye(nc+1)) * a + a.dag() * tensor(sdown_list[n],qeye(nc+1)))
#
#
#     for n in range(N - 1):
#         if n in [0,2,4,6,8]:
#             HB += -J[n] * (1+delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
#             HB += -J[n] * (1+delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))
#
#
#     for n in range(N - 1):
#         if n in [1,3,5,7]:
#             HB += -J[n] * (1-delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
#             HB += -J[n] * (1-delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))
#
#     HA += omega_c * a.dag() * a
#
#
#     HS = HA + HB + HI
#
#     # mat_1 = mat_1 / N
#     # Emax = max(HS.eigenenergies())
#     # Emin = min(HS.eigenenergies())
#     # HS = (1 / (Emax - Emin)) * (2 * HS - (Emax + Emin) * mat_1)
#     # psi_1=psi(N, J, omega_a, delta)
#     # psi_all = tensor(psi(N, J, omega_a, delta), basis(nc+1, nc))
#     # psi_all = tensor(psi(N, J, omega_a, delta), coherent(nc+1, np.sqrt(0.25)))
#     # np.savetxt('D:/python/python001/QB/data/HS2.txt', psi_1, fmt="%.4e", delimiter=" ")
#     # print(psi_all)
#     # print(psi_1)
#
#     # print(shape(HS.eigenenergies()))
#     # print(shape(psi_all))
#
#     psi_list = []
#     for n in range(N):
#         psi_list.append(basis(2, 0))
#     # psi_list.append(basis(2, 1))
#     psi0 = tensor(psi_list)
#     psi_all=tensor(psi0,coherent(nc+1, np.sqrt(2.25)))
#     E0 = expect(HB, psi_all)
#
#     def E(tlist):
#         result = mesolve(HS, psi_all, tlist, [], [HB])
#         Et = np.array(result.expect[0]) # 这里为什么要选择第一个能量？要选不应该也是-1吗？
#         return abs(Et- E0)
#     P=E(tlist)/tlist
#     return P
#     # return E(tlist)
#
# def E_fun_003(N, J, omega_a, omega_c, delta, g, tlist):
#     nc = 3 * N
#     si = qeye(2)  # Unit matrix
#     a = tensor(si, si, si, si, destroy(nc+1))
#     sup = jmat(1 / 2, '+')
#     sdown = jmat(1 / 2, '-')
#
#
#     sup_list = []
#     sdown_list = []
#     mat_list = []
#
#     for n in range(N):
#         op_list = []
#         for m in range(N):
#             op_list.append(si)
#
#         op_list[n] = sup
#         sup_list.append(tensor(op_list))
#
#         op_list[n] = sdown
#         sdown_list.append(tensor(op_list))
#
#         op_list[n] = si
#         mat_list.append(tensor(op_list))
#
#     # construct the hamiltonian
#     HS=0
#     HA=0
#     HB=0
#     HI=0
#     mat_1 = 0
#
#     for n in range(N):
#         # print(n)
#         mat_1 += tensor(mat_list[n],qeye(nc+1))
#         HB += omega_a * tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1))
#         HI += g * (tensor(sup_list[n],qeye(nc+1)) * a + a.dag() * tensor(sdown_list[n],qeye(nc+1)))
#
#
#     for n in range(N - 1):
#         if n in [0,2,4,6,8]:
#             HB += -J[n] * (1+delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
#             HB += -J[n] * (1+delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))
#
#
#     for n in range(N - 1):
#         if n in [1,3,5,7]:
#             HB += -J[n] * (1-delta) * (tensor(sup_list[n],qeye(nc+1)) * tensor(sdown_list[n+1],qeye(nc+1)))
#             HB += -J[n] * (1-delta) * (tensor(sup_list[n+1],qeye(nc+1)) * tensor(sdown_list[n],qeye(nc+1)))
#
#     HA += omega_c * a.dag() * a
#
#
#     HS = HA + HB + HI
#
#     # mat_1 = mat_1 / N
#     # Emax = max(HS.eigenenergies())
#     # Emin = min(HS.eigenenergies())
#     # HS = (1 / (Emax - Emin)) * (2 * HS - (Emax + Emin) * mat_1)
#     # psi_1=psi(N, J, omega_a, delta)
#     # psi_all = tensor(psi(N, J, omega_a, delta), basis(nc+1, nc))
#     # psi_all = tensor(psi(N, J, omega_a, delta), coherent(nc+1, np.sqrt(0.25)))
#     # np.savetxt('D:/python/python001/QB/data/HS2.txt', psi_1, fmt="%.4e", delimiter=" ")
#     # print(psi_all)
#     # print(psi_1)
#
#     # print(shape(HS.eigenenergies()))
#     # print(shape(psi_all))
#
#     psi_list = []
#     for n in range(N):
#         psi_list.append(basis(2, 0))
#     # psi_list.append(basis(2, 1))
#     psi0 = tensor(psi_list)
#     psi_all=tensor(psi0,coherent(nc+1, np.sqrt(6.25)))
#     E0 = expect(HB, psi_all)
#
#     def E(tlist):
#         result = mesolve(HS, psi_all, tlist, [], [HB])
#         Et = np.array(result.expect[0])
#         return abs(Et- E0)
#     P=E(tlist)/tlist
#     return P
#     # return E(tlist)


# tlist = np.logspace(-2.5, 0.3, 1000)
tlist = np.linspace(0.0000001, 30, 999)
N = 3  # number of spins
omega_a = 1
omega_c = 1
g = 1
delta = 0
# J=random_int_list(0,100,N)
J = 0*np.ones(N)

# 这不是一个能量，一个功率吗？为什么都有要用E开头？
E_expt1 = E_fun_001(N, J, omega_a, omega_c, delta, g, tlist)[0]
E_expt2 = E_fun_001(N, J, omega_a, omega_c, delta, g, tlist)[1]
# print(E_expt2)

# E_expt2 = E_fun_002(N, J, omega_a, omega_c, delta, g, tlist)
# E_expt3 = E_fun_003(N, J, omega_a, omega_c, delta, g, tlist)

fig = plt.figure(figsize = (10,6))
aa2 = plt.subplot(2,1,1)
# aa2.set_xscale('log')
# aa2.set_xticks([0,5,10])
# aa2.set_yticks([0,0.25,0.50,0.75])
plt.plot(tlist, E_expt1, linestyle="-.", color='green',label=r'Fock', linewidth=2.5)
plt.xlabel('$t$', fontsize=22, labelpad=1)  # 横坐标标注尺寸
plt.ylabel('$E_{t}$', fontsize=22, labelpad=1)  # 纵坐标标注尺寸
plt.axis([0, 8, 0, 6])
plt.legend(fontsize=18)
aa3 = plt.subplot(2,1,2)
# aa3.set_xscale('log')
# aa3.set_xticks([10**(-1),10**(0)])
# aa3.set_yticks([0,5,10,10])

plt.plot(tlist, E_expt2, linestyle="-.", color='mediumpurple',label=r'Fock', linewidth=2.5)
plt.axis([0, 8, 0, 10])
plt.legend(fontsize=18)
# aa4 = plt.subplot(3,1,3)
# aa4.set_xscale('log')
# aa4.set_xticks([10**(-1),10**(0)])
# aa4.set_yticks([0,5,10,15,20])
# plt.plot(tlist, E_expt3, linestyle="-.", color='mediumpurple', linewidth=2.5)
# plt.axis([0.01, 1, -0.2, 20.2])
plt.show()

# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			Entanglement,-Coherence-QB
 *Date:			2023-08-16 21:31:32
 *Description:Entanglement,Coherence and the charging process of Quantum Batteries
              ，复原这篇论文中的能量和功率的图片Fig2
"""
import numpy as np
import matplotlib.pyplot as plt


def claculate(J, triangle):
    # =======================初始化变量，态矢，能量等等===================
    Omega = 1  # 外场驱动强度
    hbar = 1  # 自然单位
    omega0 = 1  # 原子自己的震动频率
    t = np.linspace(0, np.pi / 2 / Omega, 1000)  # 定义时间序列
    # 定义一些参数
    alpha = J * (triangle - 1)  # 1
    beta = np.sqrt(J ** 2 * (triangle - 1) ** 2 + 4 * Omega ** 2)  # 2
    gamma1 = 2 * Omega / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    gamma2 = (alpha + beta) / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    E = np.ones_like(t)  # 体系能量
    C0 = np.ones_like(E) # 体系相关度
    Q = np.ones_like(E)  # 体系纠缠度

    # 单自旋的本征态
    up = np.array([[1], [0]])
    down = np.array([[0], [1]])
    # 两个粒子的本征态耦合
    uu = np.kron(up, up)
    ud = np.kron(up, down)
    dd = np.kron(down, down)
    du = np.kron(down, up)
    # 体系自由哈密顿量的矩阵表达形式
    H0 = 2 * hbar * omega0 * np.dot(uu, uu.T) + hbar * omega0 * np.dot(ud, ud.T) \
         - hbar * omega0 * np.dot(du, du.T) - 2 * hbar * omega0 * np.dot(dd, dd.T)
    # 初始化本征态
    E1_s = (dd - uu) / np.sqrt(2)
    E2_s = (du - ud) / np.sqrt(2)
    E3_s = gamma1 * (dd + uu) - gamma2 * (du + ud)
    E4_s = gamma2 * (dd + uu) + gamma1 * (du + ud)

    # 初始化本征能量
    E1 = hbar * triangle * J  # 1
    E2 = -hbar * (triangle + 2) * J  # -3
    E3 = hbar * (J - beta)  # -1
    E4 = hbar * (J + beta)  # 3

    # =======================计算体系随时间的演化===================
    # 体系初态为  dd，那么我们需要在H的表象下写出初态就需要知道每个态上的系数
    c1 = np.dot(E1_s.T, dd)
    c2 = np.dot(E2_s.T, dd)
    c3 = np.dot(E3_s.T, dd)
    c4 = np.dot(E4_s.T, dd)
    # 体系态矢随时间的演化
    Psi_t = c1 * np.kron(np.exp(-1j / hbar * E1 * t), E1_s) + c2 * np.kron(np.exp(-1j / hbar * E2 * t), E2_s) + \
            c3 * np.kron(np.exp(-1j / hbar * E3 * t), E3_s) + c4 * np.kron(np.exp(-1j / hbar * E4 * t), E4_s)
    for i in range(1000):
        # 计算密度矩阵和能量
        # 这里计算密度矩阵的时候，共轭不要忘了取
        rho = np.dot(Psi_t[:, i].reshape(4, 1), np.conj(Psi_t[:, i].reshape(1, 4)))
        E[i] = np.trace(np.dot(rho, H0)) + 2 * omega0 * hbar
        alpha_uu = np.dot(uu.T, Psi_t[:, i])
        alpha_ud = np.dot(ud.T, Psi_t[:, i])
        alpha_du = np.dot(du.T, Psi_t[:, i])
        alpha_dd = np.dot(dd.T, Psi_t[:, i])
        try:
            C0[i], Q[i] = calculate_cq(
                rho, alpha_uu, alpha_ud, alpha_du, alpha_dd
            )
        except:
            print('i={}'.format(i))

    # ========================= 计算功率 ==================
    P = np.diff(E) / (t[1]-t[0])

    return t / (np.pi / 2 / Omega), E / 4, P / 4 , C0, Q


def calculate_cq(rho, *alpha):
    a_uu, a_ud, a_du, a_dd = alpha
    c = (np.sum(np.abs(rho)) - np.trace(np.abs(rho))) / 3

    q = 2 * np.abs(a_uu * a_dd - a_ud * a_du)
    return c, q



if __name__ == "__main__":
    t, E1, P1, C1, Q1 = claculate(J=1, triangle=1)
    _, E2, P2, C2, Q2 = claculate(J=1, triangle=0)
    _, E3, P3, C3, Q3 = claculate(J=1, triangle=-1)
    fig,ax = plt.subplots(2,2)
    fig.tight_layout()

    ax[0,0].plot(t, E1, 'g--', label=r'$\Delta=1$')
    ax[0,0].plot(t, E2, 'r--', label=r'$\Delta=0$')
    ax[0,0].plot(t, E3, 'k-', label=r'$\Delta=-1$')
    ax[0,0].set_ylabel(r'$E(t) /E_{max}$')
    ax[0,0].set_xlabel(r'$t/t_{min}$')
    ax[0,0].legend()

    # plt.figure()
    ax[0,1].plot(t[0:-1], P1, 'g--', label=r'$\Delta=1$')
    ax[0,1].plot(t[0:-1], P2, 'r--', label=r'$\Delta=0$')
    ax[0,1].plot(t[0:-1], P3, 'k-', label=r'$\Delta=-1$')
    ax[0,1].set_ylabel(r'$P(t)/P_{max}$')
    ax[0,1].set_xlabel(r'$t/t_{min}$')
    ax[0,1].legend()


    ax[1,0].plot(t, Q1, 'g--', label=r'$\Delta=1$')
    ax[1,0].plot(t, Q2, 'r--', label=r'$\Delta=0$')
    ax[1,0].plot(t, Q3, 'k-', label=r'$\Delta=-1$')
    ax[1,0].set_ylabel(r'$Q$')
    ax[1,0].set_xlabel(r'$t/t_{min}$')
    ax[1,0].legend()


    ax[1,1].plot(t, C1, 'g--', label=r'$\Delta=1$')
    ax[1,1].plot(t, C2, 'r--', label=r'$\Delta=0$')
    ax[1,1].plot(t, C3, 'k-', label=r'$\Delta=-1$')
    ax[1,1].set_ylabel(r'$C_0$')
    ax[1,1].set_xlabel(r'$t/t_{min}$')
    ax[1,1].legend()
    # fig.suptitle('Fig 2')
    plt.subplots_adjust(top=0.85)
    plt.show()
    pass
# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			Entanglement,-Coherence-QB
 *Date:			2023-08-16 21:31:32
 *Description:Entanglement,Coherence and the charging process of Quantum Batteries，复原这篇论文中的能量和功率的图片
"""
import numpy as np
import matplotlib.pyplot as plt


def claculate(J, triangle):
    # =======================初始化变量，态矢，能量等等===================
    Omega = 1  # 这个不明白是什么
    hbar = 1  # 自然单位，
    omega0 = 1  # 原子自己的震动频率
    t = np.linspace(0, np.pi / 2 / Omega, 1000)  # 定义时间序列
    E = np.ones_like(t)  # 体系能量
    # 定义一些参数
    alpha = J * (triangle - 1)  # 1
    beta = np.sqrt(J ** 2 * (triangle - 1) ** 2 + 4 * Omega ** 2)  # 2
    gamma1 = 2 * Omega / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    gamma2 = (alpha + beta) / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    C0 = np.ones_like(E)
    Q = np.ones_like(E)

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
    P = np.diff(E)

    return t / (np.pi / 2 / Omega), E, P, C0, Q


def calculate_cq(rho, *alpha):
    a_uu, a_ud, a_du, a_dd = alpha
    c = (np.sum(np.abs(rho)) - np.trace(np.abs(rho))) / 3

    q = 2 * np.abs(a_uu * a_dd - a_ud * a_du)
    return c, q


if __name__ == "__main__":
    t, E1, P1, C1, Q1 = claculate(J=1, triangle=1)
    _, E2, P2, C2, Q2 = claculate(J=1, triangle=0)
    _, E3, P3, C3, Q3 = claculate(J=1, triangle=-1)
    E_max = np.max(np.array([np.max(E1), np.max(E2), np.max(E3)]))
    P_max = np.max(np.array([np.max(P1), np.max(P2), np.max(P3)]))

    plt.subplot(221)
    plt.plot(t, E1 / E_max, 'g--', label=r'$\Delta=1$')
    plt.plot(t, E2 / E_max, 'r--', label=r'$\Delta=0$')
    plt.plot(t, E3 / E_max, 'k--', label=r'$\Delta=-1$')
    plt.ylabel(r'$E(t) /E_{max}$')
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()

    # plt.figure()
    plt.subplot(222)
    plt.plot(t[0:-1], P1 / P_max, 'g--', label=r'$\Delta=1$')
    plt.plot(t[0:-1], P2 / P_max, 'r--', label=r'$\Delta=0$')
    plt.plot(t[0:-1], P3 / P_max, 'k--', label=r'$\Delta=-1$')
    plt.legend()
    plt.ylabel(r'$P(t)/P_{max}$')
    plt.xlabel(r'$t/t_{min}$')

    plt.subplot(223)
    plt.plot(t, C1, 'g--', label=r'$\Delta=1$')
    plt.plot(t, C2, 'r--', label=r'$\Delta=0$')
    plt.plot(t, C3, 'k-', label=r'$\Delta=-1$')
    plt.ylabel(r'$C_0$')
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()

    plt.subplot(224)
    plt.plot(t, Q1, 'g--', label=r'$\Delta=1$')
    plt.plot(t, Q2, 'r--', label=r'$\Delta=0$')
    plt.plot(t, Q3, 'k-', label=r'$\Delta=-1$')
    plt.ylabel(r'$Q$')
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()
    plt.show()

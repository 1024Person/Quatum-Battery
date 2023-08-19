import numpy as np
import matplotlib.pyplot as plt

# ======================== 初始化变量 ===========================、
Omega = 1  # 拉比振荡频率
omega0 = 1  # 粒子自身的振荡频率
hbar = 1  # 常量，能量单位
J = Omega  # 耦合强度
t_min = np.pi / 2 / Omega
t = np.linspace(0, t_min, 1000)


def calculate(triangle):
    # 这里面涉及到了一个变量：各向异性的参数
    # ================ 计算（初始化）体系的态
    alpha = J * (triangle - 1)
    beta = np.sqrt(J ** 2 * (triangle - 1) ** 2 + 4 * Omega * 2)
    gamma1 = 2 * Omega / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)
    gamma2 = (alpha + beta) / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)
    # 单自旋的本征态
    up = np.array([[1], [0]])
    down = np.array([[0], [1]])
    # 两个粒子的本征态耦合
    uu = np.kron(up, up)
    ud = np.kron(up, down)
    dd = np.kron(down, down)
    du = np.kron(down, up)
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
    # 定义密度矩阵
    rho = np.ones((1000, 4, 4), dtype=np.complex128)
    C = np.ones(1000)
    Q = np.ones(1000)
    for i in range(1000):
        rho[i, :, :] = np.dot(Psi_t[:, i].reshape(4, 1), np.conj(Psi_t[:, i].reshape(1, 4)))
        C[i] = 1 / 3 * (np.sum(np.abs(rho[i, :, :])) - np.trace(np.abs(rho[i, :, :])))
        alpha_uu = np.dot(np.conj(uu).T, Psi_t[:, i].reshape(4, 1))
        alpha_ud = np.dot(np.conj(ud).T, Psi_t[:, i].reshape(4, 1))
        alpha_du = np.dot(np.conj(du).T, Psi_t[:, i].reshape(4, 1))
        alpha_dd = np.dot(np.conj(dd).T, Psi_t[:, i].reshape(4, 1))
        Q[0, i] = 2 * np.abs(alpha_uu * alpha_dd - alpha_du * alpha_ud)[0]
    return (t, Q, C)


_, Q1, C1 = calculate(triangle=1)
_, Q2, C2 = calculate(triangle=0)
_, Q3, C3 = calculate(triangle=-1)

plt.subplot(121)
plt.plot(t / t_min, C1[0, :], 'g--', label=r'$\Delta=1$')
plt.plot(t / t_min, C2[0, :], 'r--', label=r'$\Delta=0$')
plt.plot(t / t_min, C3[0, :], 'k-', label=r'$\Delta=-1$')
plt.legend()
plt.ylabel(r'$C_0(t)$')
plt.xlabel(r'$t/t_{min}$')
# plt.figure()
plt.subplot(122)
plt.plot(t / t_min, Q1[0, :], 'g--', label=r'$\Delta=1$')
plt.plot(t / t_min, Q2[0, :], 'r--', label=r'$\Delta=0$')
plt.plot(t / t_min, Q3[0, :], 'k-', label=r'$\Delta=-1$')
plt.xlabel(r'$t/t_{min}$')
plt.ylabel('Q(t)')
plt.legend()
plt.show()

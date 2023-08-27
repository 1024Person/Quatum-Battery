import numpy as np
import matplotlib.pyplot as plt


def body(triangle):
    # =============================== 初始化
    Omega = 1
    hbar = 1
    omega0 = 1
    t_min = np.pi / 2 / Omega
    t_delta = t_min / 1000
    t = np.linspace(0, t_delta * 1000, 1000)
    J = Omega
    E = np.ones_like(t)
    P = np.ones_like(t)
    C0 = np.ones_like(t)
    Q = np.ones_like(t)


    alpha = J * (triangle - 1)  # 1
    beta = np.sqrt(J ** 2 * (triangle - 1) ** 2 + 4 * Omega ** 2)
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
    
    # 计算密度矩阵
    for i in range(1000):
        rho = np.dot(Psi_t[:,i].reshape(4,1),np.conj(Psi_t[:,i]).reshape(1,4))
        E[i] = np.trace(np.dot(rho,H0)) + 2*hbar*omega0
        alpha_uu = np.dot(uu.T,Psi_t[:,i])
        alpha_ud = np.dot(ud.T,Psi_t[:,i])
        alpha_du = np.dot(du.T,Psi_t[:,i])
        alpha_dd = np.dot(dd.T,Psi_t[:,i])
        C0[i],Q[i] = calculate_cq(rho,
            alpha_uu,
            alpha_ud,
            alpha_du,
            alpha_dd
        )
    P = np.diff(E) / t_delta / 4
    P_ave,C0_ave,Q_ave = calculate_ave(P,C0,Q,t_min,t_delta)
    return E[-1] / 4,P_ave,C0_ave,Q_ave


        
        
def calculate_cq(rho, *alpha):
    a_uu, a_ud, a_du, a_dd = alpha
    c = (np.sum(np.abs(rho)) - np.trace(np.abs(rho))) / 3

    q = 2 * np.abs(a_uu * a_dd - a_ud * a_du)
    return c, q

def calculate_ave(P,C0,Q,t_min,t_delta):
    P_ave = np.sum(P)*t_delta / (t_min-t_delta)
    C0_ave = np.sum(C0)*t_delta / t_min
    Q_ave = np.sum(Q)*t_delta / t_min
    return P_ave,C0_ave,Q_ave

def draw(E_ave,P_ave,Q_ave,C0_ave,triangles):
    fig = plt.figure()

    plt.plot(triangles,E_ave,'k-',label=r'$W_{fin}(\Delta)$')
    plt.plot(triangles,P_ave,'r--',label=r'$\bar{P}(\Delta)$')
    plt.plot(triangles,Q_ave,'g--',label=r'$\bar{Q}(\Delta)$')
    plt.plot(triangles,C0_ave,'b-.',label=r'$\bar{C}_{0}(\Delta)$')
    plt.xlabel(r'$\Delta$')
    plt.legend()
    plt.title('Fig 3')
    # ax.set(xticks=[-1,-0.5,0,0.5,1],xticklabels=['-1','-0.5','0','0.5','1'])
    plt.xticks([-1,-0.5,0,0.5,1])
    # plt.set_xticklabels(['-1','-0.5','0','0.5','1'])
    plt.show()


def main():
    """
    计算E,P,Q,C0的平均值，是triangle的函数
    
    """
    triangles = np.linspace(-1,1,100)
    P_ave = np.ones_like(triangles)
    C0_ave = np.ones_like(triangles)
    Q_ave = np.ones_like(triangles)
    E_ave = np.ones_like(triangles)

    for i in range(100):
        E_ave[i],P_ave[i],C0_ave[i],Q_ave[i]=body(triangles[i])
    draw(E_ave,P_ave,Q_ave,C0_ave,triangles)

if __name__ == "__main__":
    main()

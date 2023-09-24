# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			Fig4-Entanglement-Coherence-and-charging-process-of-QB
 *Date:			2023-08-22 12:44:11
 *Description:复现Fig4，这个Fig4是一个两变量函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def calculate_cq(rho, *alpha):
    """
    计算C0,Q
    """
    a_uu, a_ud, a_du, a_dd = alpha
    c = (np.sum(np.abs(rho)) - np.trace(np.abs(rho))) / 3

    q = 2 * np.abs(a_uu * a_dd - a_ud * a_du)
    return c, q


def calculate(J,triangle):
    """
    计算E,P,C0,Q的平均值
    """
    # ================== 初始化
    Omega = 1  # 表征外驱动场的驱动能力
    hbar = 1  # 自然单位，
    omega0 = 1  # 原子自己的震动频率
    t_min = np.pi / Omega / 2
    t = np.linspace(0,t_min,1000)
    t_delta = t[1] - t[0]
    alpha = J * (triangle - 1)  # 1
    beta = np.sqrt(J ** 2 * (triangle - 1) ** 2 + 4 * Omega ** 2)  # 2
    gamma1 = 2 * Omega / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    gamma2 = (alpha + beta) / np.sqrt(2 * (alpha + beta) ** 2 + 8 * Omega ** 2)  # 0.5
    E_max=4*omega0*hbar
    P_max = E_max*Omega
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
    E1_s = (dd - uu) / np.sqrt(2) # 负根号二 0 0 根号二
    E2_s = (du - ud) / np.sqrt(2) # 0 负根号二 根号二 0
    E3_s = gamma1 * (dd + uu) - gamma2 * (du + ud) # 0.5 -0.5 -0.5 0.5
    E4_s = gamma2 * (dd + uu) + gamma1 * (du + ud) # 0.5  0.5  0.5 0.5

    # 初始化本征能量
    E1 = hbar * triangle * J  # 1
    E2 = -hbar * (triangle + 2) * J  # -3
    E3 = hbar * (J - beta)  # -1
    E4 = hbar * (J + beta)  # 3

    # =======================计算体系随时间的演化===================
    # 体系初态为  dd，那么我们需要在H的表象下写出初态就需要知道每个态上的系数
    c1 = np.dot(E1_s.T, dd) # 根号二分之一
    c2 = np.dot(E2_s.T, dd) # 0
    c3 = np.dot(E3_s.T, dd) # 0.5
    c4 = np.dot(E4_s.T, dd) # 0.5
    # 体系态矢随时间的演化
    Psi_t = c1 * np.kron(np.exp(-1j / hbar * E1 * t), E1_s) + c2 * np.kron(np.exp(-1j / hbar * E2 * t), E2_s) + \
            c3 * np.kron(np.exp(-1j / hbar * E3 * t), E3_s) + c4 * np.kron(np.exp(-1j / hbar * E4 * t), E4_s)

    for i in range(1000):
        rho = np.dot(Psi_t[:,i].reshape(4,1),np.conj(Psi_t[:,i]).reshape(1,4))
        E[i] = np.trace(np.dot(rho,H0)) + 2*hbar*omega0
        alpha_uu = np.dot(uu.T, Psi_t[:, i])
        alpha_ud = np.dot(ud.T, Psi_t[:, i])
        alpha_du = np.dot(du.T, Psi_t[:, i])
        alpha_dd = np.dot(dd.T, Psi_t[:, i])
        C0[i],Q[i] = calculate_cq(
            rho,
            alpha_uu,
            alpha_ud,
            alpha_du,
            alpha_dd
        )
    
    P = np.diff(E) / t_delta / P_max
    E = E / E_max

    E_ave,P_ave,C0_ave,Q_ave = calculate_ave(E,P,Q,C0,t_min,t_delta)
    return E_ave,P_ave,C0_ave,Q_ave

def calculate_ave(E,P,Q,C0,t_min,t_delta):
    E_ave = E[-1]
    P_ave = np.sum(P) / t_min * t_delta
    C0_ave = np.sum(C0) / t_min * t_delta
    Q_ave = np.sum(Q)/t_min * t_delta
    return E_ave,P_ave,C0_ave,Q_ave

def draw(E,P,Q,C0,triangle,J):
    """
    绘制三维图像
    """
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(triangle,np.log10(J),E, cmap= cm.coolwarm)
    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('$log_{10}(J/\\Omega)$')
    ax.set_zlabel('$W_{fin}$')
    ax.set(yticks=[-1,-0.5,0,0.5,1],xticks=[-1,0,1])
    ax.set(yticklabels=['$-1$','$-0.5$','$0$','$0.5$','$1$'],
       xticklabels=['$-1$','$0$','$1$'],)
    fig.colorbar(surf, shrink=0.5, aspect=5)

# 
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(triangle,np.log10(J),P,cmap= cm.coolwarm)
    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('$log_{10}(J/\\Omega)$')
    ax.set_zlabel('$\\bar{P}$')
    ax.set(yticks=[-1,-0.5,0,0.5,1],xticks=[-1,0,1])
    ax.set(yticklabels=['$-1$','$-0.5$','$0$','$0.5$','$1$'],
       xticklabels=['$-1$','$0$','$1$'],)
    fig.colorbar(surf, shrink=0.5, aspect=5)

#  
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf= ax.plot_surface(triangle,np.log10(J),Q, cmap= cm.coolwarm)
    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('$log_{10}(J/\\Omega)$')
    ax.set_zlabel('$\\bar{Q}$')
    ax.set(yticks=[-1,-0.5,0,0.5,1],xticks=[-1,0,1])
    ax.set(yticklabels=['$-1$','$-0.5$','$0$','$0.5$','$1$'],
       xticklabels=['$-1$','$0$','$1$'],)
    fig.colorbar(surf, shrink=0.5, aspect=5)

#  
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(triangle,np.log10(J),C0, cmap= cm.coolwarm)
    ax.set_xlabel('$\\Delta$')
    ax.set_ylabel('$log_{10}(J/\\Omega)$')
    ax.set_zlabel('$\\bar{C}_{0}$')
    ax.set(yticks=[-1,-0.5,0,0.5,1],xticks=[-1,0,1])
    ax.set(yticklabels=['$-1$','$-0.5$','$0$','$0.5$','$1$'],
       xticklabels=['$-1$','$0$','$1$'],)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()




def main():
    """
    计算E,P,C0,Q的平均值作为triangle和log(J/Omega)的函数
    """
    Jn = 100
    tn = 5
    J = np.linspace(0.1,10,Jn)
    triangle = np.linspace(-1,1,tn)
    J_s,t_s = np.meshgrid(J,triangle)
    E = np.ones_like(J_s)
    P = np.ones_like(J_s)
    C0 = np.ones_like(J_s)
    Q = np.ones_like(J_s)
    # calculate(0,1)
    for i in range(tn):
        for j in range(Jn):
            E[i,j],P[i,j],C0[i,j],Q[i,j] = calculate(J_s[i,j],t_s[i,j])
            print('i={},j={}'.format(i,j))
    draw(E,P,Q,C0,t_s,J_s)


    
if __name__=="__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt



def main():
    #=======================初始化参量===================
    # hbar这里用这个数，没问题吧
    hbar = 6.626e-34
    # 这里我也不知道omega应该等于多少，就先等于1了
    Omega=1
    # 相互作用的强度参量也先定义为omega
    J=Omega
    # 和自旋链的各向异性有关的结构参数
    triangleup=1
    # 拉莫尔进动频率(这里也是我自己瞎编的，拉莫尔进动频率中有一个磁场，速度，磁导率其实是不知道的)
    omega_0 = 1
    # 最大充电能量和最大平行充电功率
    epsilon_max = 4*hbar*omega_0
    P_max_para=epsilon_max * Omega
    # 电量冲到最大值所需要的时间
    t_min_para = np.pi/(2*J)
    # 时间序列    
    t_para = np.linspace(0,t_min_para,200)
    # 单自旋的基态和激发态
    u_state=np.array([[1],[0]])
    d_state=np.array([[0],[1]])
    # 两粒子耦合之后的本征态
    uu_state = np.kron(u_state,u_state)
    dd_state=np.kron(d_state,d_state)
    ud_state=np.kron(u_state,d_state)
    du_state=np.kron(d_state,u_state)
    alpha=J*(triangleup - 1)
    beta=np.sqrt(J**2*(triangleup - 1)**2+4*Omega**2)
    gamma_1 = 2*Omega / np.sqrt(2*(alpha+beta)**2+8*Omega**2)
    gamma_2 = (alpha+beta) / np.sqrt(2*(alpha+beta)**2+8*Omega**2)
    # 充电过程的能量本征态
    E1_state=(dd_state - uu_state) / np.sqrt(2)
    E2_state=(du_state - ud_state) / np.sqrt(2)
    E3_state=(gamma_1*(uu_state+dd_state) - gamma_2*(du_state + ud_state))
    E4_state=(gamma_1*(uu_state+dd_state) + gamma_2*(du_state + ud_state))

    E1 = J*triangleup*hbar
    E2 = -J*(triangleup + 2)*hbar
    E3 = (J-beta)*hbar
    E4 = (J+beta)*hbar
    #=======================绘制平行充电的函数===================
    epsilon_para = epsilon_max * np.sin(Omega*t_para)**2
    P_para = P_max_para * np.sin(2*Omega*t_para)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(t_para/t_min_para,epsilon_para/epsilon_max,'r')
    axes[0].set_ylabel('$\epsilon_{||}/\epsilon_\max$')
    axes[0].set_xlabel('$t/t_\min$')
    axes[1].plot(t_para/t_min_para,P_para/P_max_para)
    axes[1].set_ylabel('$\mathcal{P}_{||}/\mathcal{P}_\max$')
    axes[1].set_xlabel('$t/t_\min$')
    plt.show()
    #=======================绘制集体（collective）充电的能量和功率函数===================
    # 这一部分可以先省略掉，然后回头再重新画S
    # 主要是这部分不重要，重要的是，后面的Q,C0等等之类的

    #=======================最最重要的部分：Q,C0,充电和功率的各种图像===================
    # 已经知道了一种模型了，并且知道了怎么求态等等之类的，所以可以去求解各种纠缠和相关了





if __name__ == "__main__":
    main()



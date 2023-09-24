from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tools.draw import draw_sub,draw

def claculate(triangle):
    hbar = 1
    omega0 = 1
    Omega = 1
    J = Omega
    t_min = np.pi / 2 /Omega
    t = np.linspace(0,t_min,1000)
    E = np.ones_like(t)
    P = np.ones_like(t)
    C0 = np.ones_like(t)
    Q = np.ones_like(t)
    
    up = basis(2,0)
    down = basis(2,1)
    uuu = tensor(up,up,up)
    uud = tensor(up,up,down)
    udu = tensor(up,down,up)
    duu = tensor(down,up,up)
    udd = tensor(up,down,down)
    ddu = tensor(down,down,up)
    dud = tensor(down,up,down)
    ddd = tensor(down,down,down)
    
    H0 = 3*hbar*omega0*(uuu*uuu.dag())+hbar*omega0*(uud*uud.dag()+udu*udu.dag()+duu*duu.dag())-\
            hbar*omega0*(udd*udd.dag()+dud*dud.dag()+ddu*ddu.dag())-\
            3*hbar*omega0*(ddd*ddd.dag())
    
    pauli_x_1 = tensor(sigmax(),qeye(2),qeye(2))
    pauli_x_2 = tensor(qeye(2),sigmax(),qeye(2))
    pauli_x_3 = tensor(qeye(2),qeye(2),sigmax())
    pauli_y_1 = tensor(sigmay(),qeye(2),qeye(2))
    pauli_y_2 = tensor(qeye(2),sigmay(),qeye(2))
    pauli_y_3 = tensor(qeye(2),qeye(2),sigmay())
    pauli_z_1 = tensor(sigmaz(),qeye(2),qeye(2))
    pauli_z_2 = tensor(qeye(2),sigmaz(),qeye(2))
    pauli_z_3 = tensor(qeye(2),qeye(2),sigmaz())
    
    H_ch = hbar*Omega*(pauli_x_1+pauli_x_2+pauli_x_3)
    
    H_int = J*hbar*(pauli_x_1*pauli_x_2+pauli_y_1*pauli_y_2+triangle*pauli_z_1*pauli_z_2)+\
            J*hbar*(pauli_x_2*pauli_x_3+pauli_y_2*pauli_y_3+triangle*pauli_z_2*pauli_z_3)
    H = H_ch+H_int

    Psi_t = mesolve(H,ddd,t,[],[])
    P_op = 1/1j/hbar*(H0*H_ch-H_ch*H0)
    for i in range(1000):
        rho = Psi_t.states[i]*Psi_t.states[i].dag()
        E[i] = np.trace(H0*rho)
        P[i] = np.trace(P_op*rho)
        C0[i] = (np.sum(np.abs(rho.full()))-np.trace(np.abs(rho.full()))) / 7
        rho_1 = rho.ptrace(0)
        rho_2 = rho.ptrace(1)
        rho_3 = rho.ptrace(2)
        
        Q[i] = (np.trace(rho_1*rho_1)+np.trace(rho_2*rho_2)+np.trace(rho_3*rho_3)) / 3

    return t/t_min,E / 6, P/6, C0, Q
    


def main():
    triangles=np.linspace(-1,1,5)
    style = ['k-', 'r--', 'y-.', 'b--', 'g--']
    labels = ['$\Delta=-1$', '$\Delta=-0.5$', '$\Delta=0$', '$\Delta=0.5$', '$\Delta=1$']
    E_all = []
    P_all = []
    C0_all = []
    Q_all = []
    for i in range(5):
        t, E,P, C0,Q = claculate(triangles[i])
        draw(t, E, style[i], 1, label=labels[i],ylabel= r'$W/W_{max}$')
        draw(t, P, style[i], 2, label=labels[i],ylabel= r'$P/P_{max}$')
        draw(t, C0,style[i], 3, label=labels[i],ylabel= r'C0')
        draw(t, Q,style[i], 4, label=labels[i], ylabel=r'Q')
        E_all.append(E)
        P_all.append(P)
        Q_all.append(Q)
        C0_all.append(C0)
    plt.figure(5)
    plt.subplot(221)
    plt.plot(t,E_all[0],style[0],label=labels[0])
    plt.plot(t,E_all[1],style[1],label=labels[1])
    plt.plot(t,E_all[2],style[2],label=labels[2])
    plt.plot(t,E_all[3],style[3],label=labels[3])
    plt.plot(t,E_all[4],style[4],label=labels[4])
    plt.ylabel(r'$E/E_{max}$')
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()

    plt.subplot(222)
    plt.plot(t,P_all[0],style[0],label=labels[0])
    plt.plot(t,P_all[1],style[1],label=labels[1])
    plt.plot(t,P_all[2],style[2],label=labels[2])
    plt.plot(t,P_all[3],style[3],label=labels[3])
    plt.plot(t,P_all[4],style[4],label=labels[4])
    plt.ylabel(r'$P/P_{max}$')
    plt.xlabel(r'$t/t_{min}$')
#     plt.legend()

    plt.subplot(223)
    plt.plot(t,Q_all[0],style[0],label=labels[0])
    plt.plot(t,Q_all[1],style[1],label=labels[1])
    plt.plot(t,Q_all[2],style[2],label=labels[2])
    plt.plot(t,Q_all[3],style[3],label=labels[3])
    plt.plot(t,Q_all[4],style[4],label=labels[4])
    plt.ylabel(r'$Q_{av}$')
    plt.xlabel(r'$t/t_{min}$')
#     plt.legend()

    plt.subplot(224)
    plt.plot(t,C0_all[0],style[0],label=labels[0])
    plt.plot(t,C0_all[1],style[1],label=labels[1])
    plt.plot(t,C0_all[2],style[2],label=labels[2])
    plt.plot(t,C0_all[3],style[3],label=labels[3])
    plt.plot(t,C0_all[4],style[4],label=labels[4])
    plt.ylabel(r'$C_0$')
    plt.xlabel(r'$t/t_{min}$')
#     plt.legend()


    plt.show()

if __name__ == "__main__":
    main()
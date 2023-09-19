# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			Fig5-Entanglement-Coherence-and-charging-process-of-QB
 *Date:			2023-08-23 10:32:51
 *Description:复现Fig5 这个三个单元的量子电池
"""
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
# =============================== 初始化变量
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
# =============================== 初始化算符与态
up = np.array([[1],[0]])
down = np.array([[0],[1]])

uuu = np.kron(np.kron(up,up),up)
uud = np.kron(np.kron(up,up),down)
udu = np.kron(np.kron(up,down),up)
udd = np.kron(np.kron(up,down),down)
duu = np.kron(np.kron(down,up),up)
dud = np.kron(np.kron(down,up),down)
ddu = np.kron(np.kron(down,down),up)
ddd = np.kron(np.kron(down,down),down)

H0 = 3*hbar*omega0*np.dot(uuu,uuu.T)+\
        1*hbar*omega0*(np.dot(uud,uud.T)+np.dot(udu,udu.T)+np.dot(duu,duu.T))-\
        hbar*omega0*(np.dot(udd,udd.T)+np.dot(dud,dud.T)+np.dot(ddu,ddu.T))-\
        3*hbar*omega0*np.dot(ddd,ddd.T)



# 单粒子情况下的泡利矩阵
pauli_x = np.array([[0,1],[1,0]])
pauli_y = 1j*np.array([[0,-1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

pauli_x_1 = np.kron(pauli_x,np.kron(np.eye(2),np.eye(2)))
pauli_x_2 = np.kron(np.eye(2),np.kron(pauli_x,np.eye(2)))
pauli_x_3 = np.kron(np.eye(2),np.kron(np.eye(2),pauli_x))


pauli_y_1 = np.kron(pauli_y,np.kron(np.eye(2),np.eye(2)))
pauli_y_2 = np.kron(np.eye(2),np.kron(pauli_y,np.eye(2)))
pauli_y_3 = np.kron(np.eye(2),np.kron(np.eye(2),pauli_y))


pauli_z_1 = np.kron(pauli_z,np.kron(np.eye(2),np.eye(2)))
pauli_z_2 = np.kron(np.eye(2),np.kron(pauli_z,np.eye(2)))
pauli_z_3 = np.kron(np.eye(2),np.kron(np.eye(2),pauli_z))

# 现在是三粒子状态，所以我感觉应该是改成三粒子状态
H_ch = hbar*Omega*(pauli_x_1+pauli_x_2+pauli_x_3)
def claculate(triangle):
<<<<<<< HEAD
    H_int_ =J*hbar*(np.dot(pauli_x_1,pauli_x_2)+np.dot(pauli_y_1,pauli_y_2)+triangle*np.dot(pauli_z_1,pauli_z_2)) +\
          J*hbar*(np.dot(pauli_x_2,pauli_x_3)+np.dot(pauli_y_2,pauli_y_3)+triangle*np.dot(pauli_z_2,pauli_z_3))
    H = H_ch  + H_int_
    # 定义功率算符
    P_opea = 1/1j*(np.dot(H0,H_int_)-np.dot(H_int_,H0))   # 这个无论怎么看都是0 

    engval,engstate = np.linalg.eig(H)
=======
    H_int =J*hbar*(np.dot(pauli_x_1,pauli_x_2)+np.dot(pauli_y_1,pauli_y_2)+triangle*np.dot(pauli_z_1,pauli_z_2)) +\
           J*hbar*(np.dot(pauli_x_2,pauli_x_3)+np.dot(pauli_y_2,pauli_y_3)+triangle*np.dot(pauli_z_2,pauli_z_3))
    H = H_ch  + H_int
    # 定义功率算符
    P_opea = 1/1j*(np.dot(H0,H_ch)-np.dot(H_ch,H0))   
    # 卧槽！eigh和eig这么不同吗？eig就不对，eigh就对了？啊？
    engval,engstate = np.linalg.eigh(H)
>>>>>>> eac2a5d2168db6402d73875a8b0137ea44b9ce3a

    E1=engval[0]
    E2=engval[1]
    E3=engval[2]
    E4=engval[3]
    E5=engval[4]
    E6=engval[5]
    E7=engval[6]
    E8=engval[7]

    E1_s=engstate[:,0].reshape(8,1)
    E2_s=engstate[:,1].reshape(8,1)
    E3_s=engstate[:,2].reshape(8,1)
    E4_s=engstate[:,3].reshape(8,1)
    E5_s=engstate[:,4].reshape(8,1)
    E6_s=engstate[:,5].reshape(8,1)
    E7_s=engstate[:,6].reshape(8,1)
    E8_s=engstate[:,7].reshape(8,1)
    
    c1=np.dot(np.conj(E1_s).T,ddd)
    c2=np.dot(np.conj(E2_s).T,ddd)
    c3=np.dot(np.conj(E3_s).T,ddd)
    c4=np.dot(np.conj(E4_s).T,ddd)
    c5=np.dot(np.conj(E5_s).T,ddd)
    c6=np.dot(np.conj(E6_s).T,ddd)
    c7=np.dot(np.conj(E7_s).T,ddd)
    c8=np.dot(np.conj(E8_s).T,ddd)
    
    Psi_t = c1*np.kron(np.exp(-1j*E1*t / hbar), E1_s)+\
            c2*np.kron(np.exp(-1j*E2*t / hbar), E2_s)+\
            c3*np.kron(np.exp(-1j*E3*t / hbar), E3_s)+\
            c4*np.kron(np.exp(-1j*E4*t / hbar), E4_s)+\
            c5*np.kron(np.exp(-1j*E5*t / hbar), E5_s)+\
            c6*np.kron(np.exp(-1j*E6*t / hbar), E6_s)+\
            c7*np.kron(np.exp(-1j*E7*t / hbar), E7_s)+\
            c8*np.kron(np.exp(-1j*E8*t / hbar), E8_s)
    for i in range(1000):
        rho = np.dot(Psi_t[:,i].reshape(8,1),np.conj(Psi_t[:,i]).reshape(1,8))
        E[i] = np.trace(np.dot(rho,H0)) + 3*hbar*omega0
        P[i] = np.trace(np.dot(P_opea,rho))
        # 三粒子的情况下，最大相关度变为7
        C0[i] = (np.sum(np.abs(rho))-np.trace(np.abs(rho))) / 7
    return t/t_min, E / 6,P / 6, C0

def draw(t,ags,style,fignum,label,ylabel):
    plt.figure(num=fignum)
    plt.plot(t,ags,style,label =label)
    plt.ylabel(ylabel)
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()



def main():
    triangles = [-1, -0.5, 0, 0.5, 1]
    style = ['k-', 'r--', 'y-.', 'b--', 'g--']
    labels = ['$\Delta=-1$', '$\Delta=-0.5$', '$\Delta=0$', '$\Delta=0.5$', '$\Delta=1$']
    for i in range(5):

        t, E,P, C0 = claculate(triangles[i])
        draw(t, E, style[i], 1, labels[i], r'$W/W_{max}$')
        draw(t, P, style[i], 2, labels[i], r'$P/P_{max}$')
        draw(t, C0,style[i], 3, labels[i], r'C0')
    
    plt.show()

if __name__ == "__main__":
    main()


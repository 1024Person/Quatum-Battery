import numpy as np

def calculate_coll_para(J,triangle):
    Omega = 1
    hbar = 1
    omega0 = 1
    t_min = np.pi/(2*Omega)
    t = np.linspace(0,t_min,1000)
    E = np.ones_like(t)
    P = np.ones_like(t)
    C0 = np.ones_like(t)
    Q = np.ones_like(t)

    up = np.array([[1],[0]])
    down = np.array([[0],[1]])

    uu = np.kron(up,up)
    ud = np.kron(up,down)
    du = np.kron(down,up)
    dd = np.kron(down,down)
    pauli_x = np.array([[0,1],[1,0]])
    pauli_y = 1j*np.array([[0,-1],[1,0]])
    pauli_z = np.array([[1,0],[0,-1]])

    pauli_x_1 = np.kron(pauli_x,np.eye(2))
    pauli_x_2 = np.kron(np.eye(2),pauli_x)

    pauli_y_1 = np.kron(pauli_y,np.eye(2))
    pauli_y_2 = np.kron(np.eye(2),pauli_y)

    pauli_z_1 = np.kron(pauli_z,np.eye(2))
    pauli_z_2 = np.kron(np.eye(2),pauli_z)


    H_ch = hbar*omega0*(pauli_x_1+pauli_x_2)
    H_int = J*hbar*(np.dot(pauli_x_1,pauli_x_2)+np.dot(pauli_y_1,pauli_y_2)+triangle*np.dot(pauli_z_1,pauli_z_2))
    H = H_ch+H_int

    H0 = hbar*omega0*(pauli_z_1+pauli_z_2)

    engval,engstate = np.linalg.eig(H)

    E1 = engval[0]
    E2 = engval[1]
    E3 = engval[2]
    E4 = engval[3]

    E1_s = engstate[:,0].reshape(4,1)
    E2_s = engstate[:,1].reshape(4,1)
    E3_s = engstate[:,2].reshape(4,1)
    E4_s = engstate[:,3].reshape(4,1)
    
    c1 = np.dot(np.conj(E1_s).reshape(1,4),dd)
    c2 = np.dot(np.conj(E2_s).reshape(1,4),dd)
    c3 = np.dot(np.conj(E3_s).reshape(1,4),dd)
    c4 = np.dot(np.conj(E4_s).reshape(1,4),dd)

    Psi_t = c1*np.kron(np.exp(-1j*E1*t/hbar),E1_s)+\
            c2*np.kron(np.exp(-1j*E2*t/hbar),E2_s)+\
            c3*np.kron(np.exp(-1j*E3*t/hbar),E3_s)+\
            c4*np.kron(np.exp(-1j*E4*t/hbar),E4_s)
    for i in range(1000):
        rho = np.dot(Psi_t[:,i].reshape(4,1),np.conj(Psi_t[:,i]).reshape(1,4))
        E[i] = np.trace(np.dot(rho,H0)) + 2*hbar*omega0
        C0[i] = (np.sum(np.abs(rho))-np.trace(np.abs(rho)))/3
        Q[i] = calculate_q(Psi_t[:,i].reshape(4,1),uu,ud,du,dd)

    P = np.diff(E) / (t[2]-t[1])

    return t/t_min, E/4/hbar/omega0, P / Omega/4/hbar/omega0,C0,Q

def calculate_q(Psi,*ags):
    uu,ud,du,dd = ags
    alpha_uu = np.dot(uu.T,Psi)
    alpha_ud = np.dot(ud.T,Psi)
    alpha_du = np.dot(du.T,Psi)
    alpha_dd = np.dot(dd.T,Psi)
    return 2*np.abs(alpha_uu*alpha_dd - alpha_ud*alpha_du)

if __name__ == "__main__":
    calculate_coll_para(0,1)
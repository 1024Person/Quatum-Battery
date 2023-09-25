from qutip import *
import numpy as np


def passive_e(N,H: Qobj, psis: list):
    vb1 = np.array(H.eigenenergies())

    m = []
    for psi in psis:
        psi = ptrace(psi,[i for i in range(N)])
        psi = psi*psi.dag()
        vb2 = psi.eigenenergies()
        m.append(np.sum(vb1*vb2))
    return m

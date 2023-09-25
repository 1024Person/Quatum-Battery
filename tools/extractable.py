from qutip import *
import numpy as np


def passive_e(H: Qobj, psi: Qobj):
    vb1 = np.array(H.eigenenergies())
    vb2 = psi.eigenenergies(sort='high')
    return np.sum(vb1*vb2)

import numpy as np
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			entanglement
 *Date:			2023-09-03 10:30:01
 *Description:观察当triangle = 1时集体充电和并行充电的时序图是否完全一致，
              会不会是当J不等于Omega的时候，triangle=1的情况下，集体充电并不等于并行充电
"""

from tools.entanglement import *
from tools.draw import *


def main():
    J=0
    t,E_p,P_p,*_ = calculate_coll_para(0,1)
    t,E_c,P_c,*_ = calculate_coll_para(1,1)
    draw(t,E_p,'r-',1,'Parallel charging',r'$E/E_{max}$')
    draw(t,E_c,'g--',1,'Collective charging',r'$E/E_{max}$')
    draw(t[0:-1],P_p,'r-',2,'Parallel charging',r'$P/P_{max}$')
    draw(t[0:-1],P_c,'g--',2,'Collective charging',r'$P/P_{max}$')
    plt.show()
    
    
main()



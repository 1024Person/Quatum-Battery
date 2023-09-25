# -*- coding: utf-8 -*-
"""
 *Author:			王成杰
 *Filename:			Fig3-Extractable-work-the-role-of-correlations(b)
 *Date:			2023-09-24 22:34:42
 *Description:复现Fig3的b图，a图终于是浮现成功了,复现完成这个fig之后就不再复现论文中的图片了
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# fig3中需要先知道每个量子态的N=8的最大可提取功，
# 然后，再分别对M=0……N的粒子的最大可提取功进行计算
# 将计算结果进行做比 
omega0 = 1
# tau = 



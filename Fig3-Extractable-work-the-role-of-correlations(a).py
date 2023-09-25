from qutip import *
import numpy as np
from tools.extractable import calc_ratio
import matplotlib.pyplot as plt


N = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# 画fig3（a）需要用到的数据
w1 = np.ones_like(N, dtype=float)
e1 = np.ones_like(N, dtype=float)
w2 = np.ones_like(N, dtype=float)
e2 = np.ones_like(N, dtype=float)
w3 = np.ones_like(N, dtype=float)
e3 = np.ones_like(N, dtype=float)
# 画fig3（b）需要用到的数据
wmwn1 = np.ones_like(N, dtype=float)
wmwn2 = np.ones_like(N, dtype=float)
wmwn3 = np.ones_like(N, dtype=float)
for i in range(len(N)):
    w1[i],e1[i],w2[i],e2[i], w3[i],e3[i] = calc_ratio(N[i])
for i in range(len(N)):
    wmwn1[i] = w1[i] / (w1[-1])*(7)/ (i+1)
    wmwn2[i] = w2[i] / (w2[-1])*(7)/ (i+1)
    wmwn3[i] = w3[i] / (w3[-1])*(7)/ (i+1)

we1 = w1 / (e1+0.0001)
we2 = w2 / (e2+0.0001)
we3 = w3 / (e3+0.0001)

print('计算完毕，开始画图')
# 画fig2（a）
plt.figure()
plt.scatter(N, we3, marker='s', color='green', label='squeeze')
plt.scatter(N, we1, marker='o', color='red', label='fock')
plt.scatter(N, we2, marker='^', color='blue', label='coherent')
plt.ylabel(r'$\epsilon_B^{(N)}(\bar{\tau})/E^{(N)}_B(\bar{\tau})$')
plt.xlabel(r'$N$')
plt.legend()
plt.xlim(0, 8.5)
plt.ylim(-0.1, 1.1)
plt.yticks([0,0.5, 1], ['0', r'$\frac{1}{2}$', '1'])
plt.xticks([2,4,6,8],[r'2', r'4', r'$6$', r'$8$'])
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.figure()
plt.scatter(N,wmwn3,marker='s',color='green',label='squeeze')
plt.scatter(N,wmwn1,marker='o',color='red',label='fork')
plt.scatter(N,wmwn2,marker='^',color='blue',label='coherent')
plt.ylabel(r'$\epsilon_B^{(M)}(\bar{\tau})/\epsilon_B^{(N)}(\bar{\tau})\times N / M$')
plt.xlabel(r'$M$')
plt.legend()
plt.xlim(0, 8.5)
plt.ylim(-0.1, 1.1)
plt.yticks([0,0.5, 1], ['0', r'$\frac{1}{2}$', '1'])
plt.xticks([2,4,6,8],[r'2', r'4', r'$6$', r'$8$'])
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.show()

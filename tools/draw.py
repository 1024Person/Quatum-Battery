import matplotlib.pyplot as plt

def draw(t,arg,style,fignum,label,ylabel=''):
    plt.figure(num=fignum)
    plt.plot(t,arg,style,label =label)
    plt.ylabel(ylabel)
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()
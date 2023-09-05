import matplotlib.pyplot as plt

def draw(t,arg,style,fignum,ylabel=None,label=None):
    plt.figure(num=fignum)
    if label:
        plt.plot(t,arg,style,label =label)
    else:
        plt.plot(t,arg,style)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()
import matplotlib.pyplot as plt
def draw_sub(t,arg,style,sub_num,ylabel=None,label=None):
    plt.subplot(sub_num)
    if label:
        plt.plot(t,arg,style,label =label)
    else:
        plt.plot(t,arg,style)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel(r'$t/t_{min}$')
    plt.legend()


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
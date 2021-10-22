from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt

data = normal(loc=0.0, scale=1, size=500000)

def plot_transfer_func(data, f, lims, num_bins=1000):
    ys = f(data)
    h = np.histogram(ys, num_bins, density=False)

    # plot output
    plt.subplot(2, 2, 1)
    plt.plot(h[0], h[1][1:], lw=4)
    plt.ylim(lims)
    plt.gca().xaxis.set_ticklabels([])
    plt.title('output')
    plt.axhline(np.mean(ys), ls='--', lw=2)

    # plot transfer function
    plt.subplot(2, 2, 2)
    x = np.arange(lims[0], lims[1], 0.1)
    y = f(x)
    plt.plot(x, y)
    isct = f(0)
    plt.plot([0, 0, lims[0]], [lims[0], isct, isct], c='r')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title('transfer function')

    # plot input
    h = np.histogram(data, num_bins, density=True)
    plt.subplot(2, 2, 3)
    plt.plot(h[1][1:], h[0], lw=4)
    plt.xlim(lims)
    plt.gca().yaxis.set_ticklabels([])
    plt.title('input')

    plt.show()


def g(x):
    return (np.cos(4*(x/2+0.7)))-1.3*x

plot_transfer_func(data, g, lims=(-3.5, 3.5), num_bins=300)
from matplotlib.patches import Ellipse, Arrow
import matplotlib.pyplot as plt
import numpy as np

def arrow(x1, y1, x2, y2, width=0.2):
    return Arrow(x1, y1, x2-x1, y2-y1, lw=1, width=width, ec='k', color='k')

ax = plt.gca()

ax.add_artist(Ellipse(xy=(2, 5), width=2, height=3, angle=70, linewidth=1, ec='k'))
ax.add_artist(Ellipse(xy=(7, 5), width=2.2, alpha=0.3, height=3.8, angle=150, linewidth=1, ec='k'))

ax.add_artist(arrow(2, 5, 6, 4.8))
ax.add_artist(arrow(1.5, 5.5, 7, 3.8))
ax.add_artist(arrow(2.3, 4.1, 8, 6))

ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)

plt.axis('equal')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
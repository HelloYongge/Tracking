from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

ax = plt.gca()
ax.add_artist(Ellipse(xy=(2, 5), alpha=0.5, width=2, height=3, angle=0.1, linewidth=1, ec='k'))
ax.add_artist(Ellipse(xy=(5, 5), alpha=0.5, width=2, height=3, angle=0.1, linewidth=1, ec='k'))
ax.add_artist(Ellipse(xy=(8, 5), alpha=0.5, width=2, height=3, angle=0.1, linewidth=1, ec='k'))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

plt.scatter([1.5, 2, 2.5], [5, 5, 5], c='k', s=50)
plt.scatter([2, 2], [4.5, 5.5], c='k', s=50)

plt.scatter([4.8, 5, 5.2], [5, 5, 5], c='k', s=50)
plt.scatter([5, 5], [4.8, 5.2], c='k', s=50)

plt.scatter([7.2, 8, 8.8], [5, 5, 5], c='k', s=50)
plt.scatter([8, 8], [4, 6], c='k', s=50)

plt.axis('equal')
plt.xlim(0, 10)
plt.ylim(1, 10)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

xs = np.arange(-4, 4, 0.1)

# 标准正态分布
ys = norm.pdf(xs)
samples = [0, 1.2, -1.2]

# 生成对称的三个点
for x in samples:
    plt.scatter([x], norm.pdf(x), s=80)

plt.plot(xs, ys)
plt.show()


import numpy as np

# X_aug = np.zeros(7)
# print(X_aug)
#
# new = 0
#
# for i in range(5):
#     new += i
#     print(new)
#
# Zmean = np.zeros((3, 1))
# b = np.array(3)
# print(np.shape(Zmean[0:2, 0:1]))
# print(type(Zmean))
# print(type(b))
#
# print(Zmean)
# print(b)
#
# print(Zmean.flatten())
#
# print(Zmean.flatten()[0])

# import numpy as np
#
# x = np.array([[[0], [1], [2]]])
#
# print(x.shape)
#
# print(x.squeeze(2).shape)

# a = np.array((1, 2, 3))
# b = np.array((4, 5, 6))
# print(np.shape(a))
# print(np.shape(b))
# print(np.dot(a, b))
# print(np.shape(np.dot(a, b)))
# c = np.reshape(a, (a.shape[0], 1))
# d = np.reshape(b, (b.shape[0], 1))
# print(np.shape(c))
# print(np.shape(d))
# print(np.dot(c, d.T))
# print(np.matmul(a, b))
# print(np.matmul(c, d.T))

import numpy
import scipy.linalg

numpy.random.seed(0)
X = numpy.random.normal(size=(10, 4))
V = numpy.dot(X.transpose(), X)
R = V.copy()
VV = scipy.linalg.cholesky(R)
print(VV)
print(R)
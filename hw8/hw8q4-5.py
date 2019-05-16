from matplotlib import pyplot as plt
import numpy as np
from numpy import sin
from numpy import cos
from math import pi

fpath = '/Users/rex/python/COMP_py/hw8/hw8.csv'
data = np.genfromtxt(fpath, delimiter=',')
a, b = data.shape
x1 = data[:,:500]
x2 = data[:,500:1000]
sigma = data[:,1000]

n = 1000
w = np.random.normal(size=(500, n))


k_ac = 1 / (2*pi) * (sin(sigma)+(pi-sigma)*cos(sigma))

k_approx = []
for row in range(500):
    temp_1 = x1[row]
    temp_2 = x2[row]
    sum_ = 0
    for i in range(n):
        max_1 = max(0, np.inner(w[:,i], temp_1))
        max_2 = max(0, np.inner(w[:,i], temp_2))
        sum_ += max_1 * max_2
    k_approx.append(sum_/n)

plt.figure()
plt.xlabel('sigma')
plt.ylabel('k')
plt.title('n='+str(n))
plt.plot(sigma, k_ac, label='k_ac')
plt.plot(sigma, k_approx, '.', label='k_approx')
plt.legend()
plt.show()




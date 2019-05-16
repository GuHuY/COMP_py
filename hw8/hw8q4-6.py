from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(-10, 10, 2000)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2*sigmoid(2*x) - 1

def relu_(x):
    if x < 0: return 0
    else: return x

def relu(x):
    output = []
    for val in x:
        output.append(relu_(val))
    return np.array(output)


n = 1000
w = np.random.normal(size=n)

y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

k_approx_sigmoid = []
k_approx_tanh = []
k_approx_relu = []
k_approx_self = []
for row in range(len(x)):
    temp_1 = x[row]
    temp_2 = y_sigmoid[row]
    temp_3 = y_tanh[row]
    temp_4 = y_relu[row]
    sum_sigmoid = 0
    sum_tanh = 0
    sum_relu = 0
    sum_self = 0
    for i in range(n):
        max_1 = max(0, w[i]*temp_1)
        max_2 = max(0, w[i]*temp_2)
        max_3 = max(0, w[i]*temp_3)
        max_4 = max(0, w[i]*temp_4)
        sum_sigmoid += max_1 * max_2
        sum_tanh += max_1 * max_3
        sum_relu += max_1 * max_4
        sum_self += max_1 * max_1
    k_approx_sigmoid.append(sum_sigmoid/n)
    k_approx_tanh.append(sum_tanh/n)
    k_approx_relu.append(sum_relu/n)
    k_approx_self.append(sum_self/n)


plt.figure()
plt.xlabel('x')
plt.ylabel('k')
plt.title('n='+str(n))
# plt.plot(x, y, '.', label='k_approx')
# plt.plot(x, k_approx_sigmoid-y_sigmoid, '.', label='K_sigmoid-sigmoid')
# plt.plot(x, k_approx_tanh-y_tanh, '.', label='K_tanh-tanh')
plt.plot(x, k_approx_relu-y_relu, '.', label='K_relu-relu')
plt.plot(x, k_approx_self-x, '.', label='K_self-x')
plt.legend()
plt.show()




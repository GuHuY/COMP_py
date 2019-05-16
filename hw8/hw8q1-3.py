from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np

clase_1 = [[1, 1], [1, 2], [2, 1]]
clase_2 = [[0, 0], [0, 1], [1, 0]]
len_class_1 = len(clase_1)
len_class_2 = len(clase_2)
X = np.array(clase_1+clase_2)
y = [-1] * len_class_1 + [1] * len_class_2

clf = SVC(kernel='linear', C=10000)
clf.fit(X, y)

w = clf.coef_[0]
k = -(w[0]/w[1])
b = -(clf.intercept_[0]/w[1])
xx = np.linspace(0, 2)
yy = k * xx + b

v_up = clf.support_vectors_[-1]
v_down = clf.support_vectors_[0]
yy_up = k * xx + (v_up[1] - k * v_up[0])
yy_down = k * xx + (v_down[1] - k * v_down[0])
margin = 1 / np.linalg.norm(w/2)

print('k =',k)
print('b =',b)
print('vectors:\n', clf.support_vectors_)
print('margin =', margin)
# print('clf.coef_:',clf.coef_)
# print('clf.intercept_:',clf.intercept_)

new_data = np.array([[7.23, 5.14]])
distance = clf.decision_function(new_data)/np.linalg.norm(w)
print('distance =', abs(distance))


plt.plot(xx, yy, label='hyperplane')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.plot(X[:len_class_1, 0], X[:len_class_1, 1], 'rx', label='class 1')
plt.plot(X[len_class_1:, 0], X[len_class_1:, 1], 'g+', label='class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.axis('tight')
plt.show()

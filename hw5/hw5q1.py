from sklearn.decomposition import PCA
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio


cifar10 = sio.loadmat('cifar10_data_batch_1.mat')
batch_label = cifar10['batch_label']
labels = cifar10['labels']
data = cifar10['data']

Class_indexes = []
for i in range(10):
    Class_indexes.append(np.where(labels==i)[0].tolist())

Indexes_sum = Class_indexes[0] + Class_indexes[1] + Class_indexes[6]
Seleted_data = data[Indexes_sum]
Seleted_labels = labels[Indexes_sum].flatten()

l1 = len(Class_indexes[0])
l2 = l1 + len(Class_indexes[1])

my_pca = PCA(n_components=2, copy=True, whiten=True)
my_pca.fit(Seleted_data)

Projected = my_pca.transform(Seleted_data )

plt.figure()
plt.scatter(Projected[:l1, 0], Projected[:l1, 1], c='r', edgecolor='none', alpha=0.9, s=12, label='0')
plt.scatter(Projected[l1:l2, 0], Projected[l1:l2, 1], c='g', edgecolor='none', alpha=0.9, s=12, label = '1')
plt.scatter(Projected[l2:, 0], Projected[l2:, 1], c='b', edgecolor='none', alpha=0.9, s=12, label = '6')
plt.legend()
plt.show()

pass
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import pandas as pd

cifar10 = sio.loadmat('cifar10_data_batch_1.mat')
batch_label = cifar10['batch_label']
labels = cifar10['labels']
data = cifar10['data']

# Select three classes
# Seleced_classes_indexes = [i for i,value in enumerate(labels) if value in [1, 2, 3]]
# result = pd.value_counts(labels)
Class_indexes = []
for i in range(10):
    Class_indexes.append(np.where(labels==i)[0].tolist())

Indexes_sum = []
for i in range(10):
    Indexes_sum = Indexes_sum + Class_indexes[i]
Seleted_data = data[Indexes_sum]
Seleted_labels = labels[Indexes_sum].flatten()

def label_transformation(old_label, extend = None):
    """
    Parameter:
        old_label(list/numpy.array): 1D list / shape(n,1) array of labels.
    Returns:
        Two_D_labels_list(numpy.array): 2D labels_list.
        dict(dictonary): Frequency of each label kind.
    """
    old_label = old_label.tolist()
    dict = {}
    for key in old_label:
        dict[key] = dict.get(key, 0) + 1
    key_list = list(dict.keys())
    key_list.sort() 
    sorted_dict = {}
    key_index = 0
    for sorted_key in key_list:
        sorted_dict[sorted_key] = key_index
        key_index = key_index + 1
    Two_D_labels_list = np.zeros((len(old_label), key_index), dtype=np.int)
    colors = []
    aim_colors = []
    colors.extend(mpl.cm.jet(np.linspace(0, 1, key_index)))
    for i in range(len(old_label)):
        j = sorted_dict[old_label[i]]
        Two_D_labels_list[i][j] = 1 
        aim_colors.append(colors[j])
    if not extend:
        return Two_D_labels_list
    else:
        if extend == 'stat':
            return dict
        if extend == 'color':
            return aim_colors, colors

my_pca = PCA(n_components=2)
my_pca.fit(Seleted_data)
# Projected = my_pca.transform(Seleted_data)
# print(my_pca.explained_variance_ratio_)
# print(my_pca.explained_variance_)

# Seleted_labels_2D = label_transformation(Seleted_labels)
# (Seleted_labels_color, colors)= label_transformation(Seleted_labels, 'color')


# plt.figure()
# for i in range(10):
#     plt.text(Projected[i, 0], Projected[i, 1], str(i), color=plt.cm.Set1(i), 
#              fontdict={'weight': 'bold', 'size': 9})
# plt.show()


Indexes_sum = Class_indexes[0] + Class_indexes[1] + Class_indexes[6]
Seleted_data_2 = data[Indexes_sum]
Seleted_labels_2 = labels[Indexes_sum].flatten()
my_pca_2 = PCA(n_components=2)
my_pca.fit(Seleted_data_2)
my_pca_2.fit(Seleted_data_2)

Projected_2 = my_pca.transform(Seleted_data_2)

Seleted_labels_2D_2 = label_transformation(Seleted_labels_2)
plt.figure()
plt.scatter(Projected_2[:, 0], Projected_2[:, 1], c=Seleted_labels_2D_2, edgecolor='none',
            alpha=0.7, s=12, cmap=plt.cm.get_cmap('Spectral'))

plt.show()




# fig = plt.figure()
# l1 = len(Class_1_indexes)
# l2 = l1 + len(Class_2_indexes)
# plt.scatter(Projected[:l1, 0], Projected[:l1, 1] ,marker='.')
# plt.scatter(Projected[l1:l2, 0], Projected[l1:l2, 1] ,marker='.')
# plt.scatter(Projected[l2:, 0], Projected[l2:, 1] ,marker='.')
# plt.show()


# pca = PCA(n_components=3)
# Projected = pca.fit(Seleted_data)
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(Projected[:l1, 0], Projected[:l1, 1], Projected[:l1, 2], marker='.')
# plt.scatter(Projected[l1:l2, 0], Projected[l1:l2, 1], Projected[l1:l2, 2], marker='.')
# plt.scatter(Projected[l2:, 0], Projected[l2:, 1], Projected[l2:, 2], marker='.')
# plt.show()

pass
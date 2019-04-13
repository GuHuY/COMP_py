from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

X = np.array([[1, 3], [1, 4], [5, 2], [5, 1], [2, 2], [7, 2]])
label = ['a', 'b', 'c', 'd', 'e', 'f']
D1 = linkage(X, 'single', metric='euclidean')
D2 = linkage(X, 'single', metric='cityblock')
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Dendrogram by euclidean')
dendrogram(D1, leaf_rotation=0, labels=label)
plt.subplot(1, 2, 2)
plt.title('Dendrogram by city-block ')
dendrogram(D2, leaf_rotation=0, labels=label)
plt.show()

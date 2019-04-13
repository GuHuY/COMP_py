from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time 
from sklearn.preprocessing import StandardScaler


# config InlineBackend.figure_format = "svg"

cifar10 = sio.loadmat('cifar10_data_batch_1.mat')
labels = cifar10['labels']
data = cifar10['data']


input_label = labels
input_data = StandardScaler().fit_transform(data)

print('TSNE...', end="")
m_time = time.time()
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(input_data)
m_time = int(time.time() - m_time)

print(str(m_time))
print('PCA...', end="")
m_time = time.time()

X_pca = PCA(n_components=2, whiten=True ).fit_transform(input_data)
# svd_solver='randomized',  random_state=33, iterated_power=100
m_time = int(time.time() - m_time)
print(str(m_time))
# X_pca = np.load("X_pca.npy")
# X_tsne = np.load("X_tsne.npy") 
font = {"color": "darkred",
        "size": 13, 
        "family" : "serif"}

plt.style.use("dark_background")
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=input_label.flatten().tolist(), alpha=0.6, s=5,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.title("t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=range(10)) 
cbar.set_label(label='Class Index', fontdict=font)
plt.clim(-0.5, 9.5)
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=input_label.flatten().tolist(), alpha=0.6, s=5,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.title("PCA", fontdict=font)
cbar = plt.colorbar(ticks=range(10)) 
cbar.set_label(label='Class Index', fontdict=font)
plt.clim(-0.5, 9.5)
plt.tight_layout()
plt.show()
pass
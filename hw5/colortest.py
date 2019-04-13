import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

N = 10

np.random.seed(10)    
a=np.random.randint(-1, N, size=(5, 5))
print(a)

fig, ax = plt.subplots()
colors = []
colors.extend(mpl.cm.jet(np.linspace(0, 1, N)))
cmap = mpl.colors.ListedColormap(colors)
mat=ax.matshow(a, cmap=cmap, vmin=0, vmax=N)
cax = plt.colorbar(mat, ticks=np.linspace(0-0.5, N-0.5, N))
cax.set_ticklabels(range(0, N))
plt.show()

pass
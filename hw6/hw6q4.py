from matplotlib import pyplot as plt
import numpy as np
X = np.linspace(-5, 8, 100, endpoint=False)
Y = np.exp(X)/(np.exp(X)+2*np.exp(0.2))
plt.figure()
plt.title('sigma3')
plt.plot(X, Y)
plt.show()



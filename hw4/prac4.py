import numpy as np, os, random, shutil
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
np.seterr(divide='ignore', invalid='ignore')

seed  = 1
np.random.seed(seed); random.seed(seed)

def main():
    print('generating or loading data...')
    # n = 300
    # k_ref = 4
    # X, _ = make_blobs(n_samples = n, n_features=2, centers=k_ref, cluster_std=1)
    # plt.figure()
    # plt.plot(X[:,0], X[:,1],'o',markerfacecolor='none',markeredgecolor='black',markersize = 5)
    # plt.show()

    fpath = '/Users/rex/COMP_py/heightWeight.csv'
    data = np.genfromtxt(fpath, delimiter=',')
    X = data[:, 1:]
    y = data [:,0]
    _ = data[:,0]
    n = X.shape[0]
    k_ref = len(set(y))

    print('clustering...')
    k = k_ref
    n_iter = 40
    m = initialize_cluster_center(k,X,'assign_to_k_random_data_point')
    m_history = [np.copy(m)]
    b_history = [np.zeros((n,k))]

    for _ in range(1, n_iter+1):
        b = np.zeros((n, k))
        for t in range(n):
            i = None
            prev_distance = np.infty
            for j in range(k):
                distance = get_distance(X[t], m[j], 'euclidean')
                if distance  < prev_distance:
                    i = j
                    prev_distance = distance
            b[t][i] = 1

        for i in range(k):
            numer = 0.
            denum = 0.
            for t in range(n):
                numer += b[t][i] * X[t]
                denum += b[t][i]
            m[i] = numer / denum

            m_history.append(np.copy(m))
            b_history.append(np.copy(b))

    print('plotting...')
    m_history = np.asarray(m_history)
    dirpath = os.path.join('plot', 'kmeans')
    figformat = 'png'
    if os.path.exists(dirpath): 
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    for h in range(n_iter+1):
        fig, ax = plt.subplots(figsize = (8, 6))
        fname = 'kmeans_' + str(h).zfill(3)+'.'+figformat
        if h == 0:
            plt.plot(X[:,0], X[:,1],'o',markerfacecolor='none',markeredgecolor='black',markersize = 5)
        else:
            color = {0:'red', 1:'green', 2:'blue', 3:'magenta'}
            for t in range(n):
                b_list = [int(i) for i in b_history[h][t].tolist()]
                cluster_idx = b_list.index(1)
                plt.plot(X[t,0], X[t,1], 'o', color  = color[cluster_idx], markersize = 5)
        plt.plot(m_history[h,:,0],m_history[h,:,1],'x',color = 'black')
        plt.xlabel('1st feature demension')
        plt.ylabel('2nd feature dimension')
        plt.title('iter= '+str(h).zfill(2))
        plt.savefig(os.path.join(dirpath,fname),dpi = 300, format = figformat, bbox_inches = 'tight')
        plt.close(fig)
    pass
    

def get_distance(x, y, distance_id):
    if distance_id  == 'euclidean':
        return np.sqrt(np.sum((x-y)**2))
    else:
        raise NotImplementedError


def initialize_cluster_center(k, X, variant):
    n = X.shape[0]
    m = []
    if variant == 'assign_to_k_random_data_point':
        idx_list = np.random.choice(n,k)
        for i in range(k):
            m_i = X[idx_list[i]]
            m.append(m_i)
    else:
        raise NotImplementedError
    return m


if __name__ == "__main__":
    main()

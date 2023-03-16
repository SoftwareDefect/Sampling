#CNN

from sklearn.neighbors import DistanceMetric
import numpy as np


def condensed_nn(X, y):
    S_fea = X
    S_idx = [i for i in range(1, X.shape[0])]
    U = [0]

    dist = DistanceMetric.get_metric('euclidean')

    for i in range(X.shape[0]):
        prev_len_u = len(U)

        n_s_idx = np.array(S_idx)
        n_u_idx = np.array(U)
        n_d = dist.pairwise(S_fea[n_s_idx], S_fea[n_u_idx])

        for i in range(n_d.shape[0]):
            nearest_u_idx = n_u_idx[n_d[i].argsort()[0]]
            p_idx = n_s_idx[i]
            if y[nearest_u_idx] != y[p_idx]:
                U.append(p_idx)
                S_idx.remove(p_idx)
                break
        if prev_len_u == len(U):
            break

    n_U = np.array(list(set(U + [i for i in range(y.shape[0]) if y[i]==1])))
    n_X, n_y = X[np.array(n_U)], y[np.array(n_U)]
    # print('before: n_pos= %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    # print('after: n_pos= %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y
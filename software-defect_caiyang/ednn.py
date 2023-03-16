#ENN
import numpy as np
from sklearn.neighbors import DistanceMetric

def edited_nn(X, y, k, isthrd=True, islog=True):
    thrd = 1 if not isthrd else k/2
    neg = X[y==0]
    dist = DistanceMetric.get_metric('euclidean')
    n_d = dist.pairwise(neg, X)
    top_k_idx = n_d.argsort()[:,:k+1]
    top_k_label = np.array([y[top_k_idx[i]] for i in range(top_k_idx.shape[0])])
    n_X = np.vstack((neg[np.sum(top_k_label, axis=1)<thrd], X[y==1]))
    n_y = np.array([0]*np.sum(np.sum(top_k_label, axis=1)<thrd) + [1]*np.sum(y==1))

    print('before: n_pos= %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos= %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y = edited_nn(trn_X, trn_y, k=15)
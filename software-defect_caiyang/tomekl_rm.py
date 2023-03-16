#Tomek Link Removal
import numpy as np
from sklearn.neighbors import DistanceMetric


def tomek_link_rm(X, y):
    dist = DistanceMetric.get_metric('euclidean')
    n_d = dist.pairwise(X, X)
    top_k_idx = n_d.argsort()[:,:2]
    tomek_link_idx = [[top_k_idx[i][0], top_k_idx[i][1]] for i in range(top_k_idx.shape[0]) if np.sum(y[top_k_idx[i]])==1]
    rm_pos_idx = [idx for idx in set([pair[0] for pair in tomek_link_idx] + [pair[1] for pair in tomek_link_idx]) if y[idx]==0]
    print('tomek_link pairs = %d, removed pos samples = %d'%(len(tomek_link_idx), len(rm_pos_idx)))
    retained_flag = np.array([1]*X.shape[0])
    retained_flag[np.array(rm_pos_idx)]=0

    n_X = X[retained_flag==1]
    n_y = y[retained_flag==1]

    return n_X, n_y


    print('before: n_pos= %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos= %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))

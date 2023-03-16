##
import numpy as np
from sklearn.neighbors import DistanceMetric

def near_miss(X, y, k, balanced_ratio=0.5):

    pos = X[y==1]
    neg = X[y==0]

    n_sample = int(pos.shape[0]/balanced_ratio)

    dist = DistanceMetric.get_metric('euclidean')
    n_d = dist.pairwise(neg, pos)
    # aa = np.array([[1], [3]])
    # bb = np.array([[5], [7]])
    # aaaa = dist.pairwise(aa, bb)
    # pairwise点对点距离  aaaa输出结果是[4(1到5的距离),6(1到7的距离)]，[2(2到5的距离)，4(3到7的距离)]，但是并不是全部
    top_k_idx = n_d.argsort()[:,:k]
    dist_2_k_pos = np.mean([n_d[i][top_k_idx[i]] for i in range(neg.shape[0])], axis=1)
    sampled_S = neg[dist_2_k_pos.argsort()[:n_sample]]

    n_X = np.vstack((sampled_S, pos))
    n_y = np.array([0]*sampled_S.shape[0] + [1]*pos.shape[0])
    print('before: n_pos= %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos= %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y =near_miss(trn_X, trn_y, k=3, balanced_ratio=0.5)
#随即上采样
import numpy as np


def random_oversampling(X, y, balanced_ratio=0.5):
    '''
        balanced_ratio = n_pos/n_neg
    '''
    r = np.sum(y==1)/np.sum(y==0)
    if r > balanced_ratio:
        return X, y

    n_over_pos = int(np.sum(y==0)*balanced_ratio - np.sum(y==1))
    pos_idx = [i for i in range(y.shape[0]) if y[i]==1]
    sampled_idx = np.random.choice(pos_idx, n_over_pos)
    n_X = np.vstack((X, X[sampled_idx]))
    n_y = np.hstack((y, y[sampled_idx]))

    print('before: n_pos = %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos = %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y = random_oversampling(X=trn_X, y=trn_y, balanced_ratio=0.5)
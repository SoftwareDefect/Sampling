#随即下采样
import random
import numpy as np


def random_undersampling(X, y, balanced_ratio):
    '''
        balanced_ratio = n_pos/n_neg
    '''
    print(type(y))
    ratio = (np.sum(y==1)/np.sum(y==0))/balanced_ratio

    print('rario= %d'%( ratio))
    n_under_sampling_idx = np.array([0 if y[i]==0 and random.random()>ratio else 1 for i in range(y.shape[0])])
    print("n_under_sampling_idx is :")
    print(n_under_sampling_idx)
    n_X = X[n_under_sampling_idx==1]
    n_y = y[n_under_sampling_idx==1]
    print('before: n_pos= %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos= %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y = random_undersampling(trn_X, trn_y, balanced_ratio=0.5)
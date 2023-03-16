#SMOTE + Tomek link removal
import numpy as np
from tomekl_rm import tomek_link_rm
from bSmote import borderline_smote

def combine_smote_tomek(X, y, balanced_ratio):
    # 采用Tomek Link Removal方法对多数类样本进行下采样
    n_X, n_y = tomek_link_rm(X, y)
    # 采用bordline SMOTE方法来对正样本上采样
    if np.sum(n_y==1)<np.sum(n_y==0) :
      n_X, n_y = borderline_smote(n_X, n_y, balanced_ratio=balanced_ratio)
    print('before: n_pos = %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos = %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y = combine_smote_tomek(trn_X, trn_y, balanced_ratio=0.25)
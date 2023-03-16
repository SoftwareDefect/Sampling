#BalanceCascade
import numpy as np

def BalanceCascade(trn_X, trn_y, tst_X, tst_y, clf, n_iter):
    neg_num = np.sum(trn_y == 0)
    pos_num = np.sum(trn_y == 1)
    neg_idx = np.argwhere(trn_y == 0).reshape(neg_num, )
    pos_idx = np.argwhere(trn_y == 1).reshape(pos_num, )
    pos_trn = trn_X[pos_idx]

    FP = pow(pos_num/neg_num, 1/(n_iter-1))
    classifiers = {}
    thresholds = {}
    tst_prob = np.empty((tst_X.shape[0], n_iter))
    for i in range(n_iter):
        classifiers[i] = clf()
        neg_trn_idx = np.random.permutation(neg_idx)[:pos_num]
        neg_trn = trn_X[neg_trn_idx]
        sub_X = np.vstack((pos_trn, neg_trn))
        sub_y = np.array([1]*pos_trn.shape[0] + [0]*neg_trn.shape[0])
        classifiers[i].fit(sub_X, sub_y)
        trn_pred = classifiers[i].predict_proba(trn_X[neg_idx])[:,1]

        thresholds[i] = np.sort(trn_pred)[int(neg_idx.shape[0]*(1-FP))]
        neg_idx = np.argwhere(trn_pred >= thresholds[i]).reshape(-1, )
        tst_prob[:,i] = classifiers[i].predict_proba(tst_X)[:,1] + thresholds[i]

    ensemble_pred = np.average(tst_prob, axis=1)
    return ensemble_pred
  #  p_r, n_p = evaluate(tst_y, ensemble_pred)
    #print('[%s] Precison on Majarity: %.3f, Recall on Minority: %.3f'%(clf.__name__, n_p, p_r))

    #我们可以分别采用如下三种分类器作为Emsemble算法的基分类器：
    #BalanceCascade(trn_X, trn_y, tst_X, tst_y, clf=LogisticRegression, n_iter=20)
    #BalanceCascade(trn_X, trn_y, tst_X, tst_y, clf=GradientBoostingClassifier, n_iter=20)
    #BalanceCascade(trn_X, trn_y, tst_X, tst_y, clf=RandomForestClassifier, n_iter=20)
#EasyEnsemble

import numpy as np



def EasyEnsemble(trn_X, trn_y, tst_X, tst_y, clf, n_iter):
    num_pos = np.sum(trn_y)
    trn_pos = trn_X[trn_y==1]
    trn_neg = trn_X[trn_y==0]
    neg_idx = list(range(trn_neg.shape[0]))

    tst_pred = np.empty((tst_X.shape[0], n_iter))
    trn_pred = np.empty((trn_X.shape[0], n_iter))

    for i in range(n_iter):
        classifier = clf()
        # Sample the same number of negative cases as positive cases
        # neg_sample = trn_neg.sample(num_pos, random_state = i)
        sampled_neg_idx = np.random.choice(neg_idx, num_pos)
        neg_sample = trn_neg[sampled_neg_idx]
        x_sub = np.vstack((neg_sample, trn_pos))
        y_sub = np.array([0]*neg_sample.shape[0] + [1]*trn_pos.shape[0])

        # Fit the classifier to the balanced dataset
        classifier.fit(x_sub, y_sub)
        pred = classifier.predict_proba(tst_X)[:,1]
        tst_pred[:, i] = pred

    # Average all the test predictions
    ensemble_pred = np.mean(tst_pred, axis = 1)
    return ensemble_pred
   # p_r, n_p = evaluate(tst_y, ensemble_pred)
   # print('[%s] Precison on Majarity: %.3f, Recall on Minority: %.3f'%(clf.__name__, n_p, p_r))
#  可使用以下三种分类器
   # EasyEnsemble(trn_X, trn_y, tst_X, tst_y, clf=LogisticRegression, n_iter=20)
    #EasyEnsemble(trn_X, trn_y, tst_X, tst_y, clf=GradientBoostingClassifier, n_iter=20)
  #  EasyEnsemble(trn_X, trn_y, tst_X, tst_y, clf=RandomForestClassifier, n_iter=20)

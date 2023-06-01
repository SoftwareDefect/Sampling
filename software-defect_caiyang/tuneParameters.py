from hpsklearn import HyperoptEstimator, standard_scaler, logistic_regression, random_forest_classifier
from hyperopt import hp, tpe
# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def fn_AUC(*params):
    auc = roc_auc_score(*params)
    return 1 - auc


def lossfn(true, pre):
    return 1 - roc_auc_score(true, pre)


def tuneclassifier(X, y, cla):
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    if cla == "LR":
        lr = logistic_regression("lr", penalty=hp.choice("LR_penalty", ["l1", "l2"]),
                                 C=hp.choice("LR_C", [1, 2, 3, 4, 5]),
                                 solver='liblinear')  # , l1_ratio=hp.choice("LR_l1_ratio", [0.5])
        # lr = logistic_regression("lr", penalty=hp.choice("LR_penalty", ["l1", "l2"]),
        #                          C=hp.uniform('LR_C', 0, 5), solver='liblinear')
        estim = HyperoptEstimator(classifier=lr, preprocessing=[standard_scaler("myZscore")], algo=tpe.suggest,
                                  loss_fn=fn_AUC, max_evals=10, n_jobs=8)  # , n_jobs=4
    else:
        rf = random_forest_classifier("rf", max_depth=hp.choice("rf_max_depth", range(1, 10)),
                                      n_estimators=hp.choice("rf_n_estimators", range(50, 300, 50)))
        # rf = random_forest_classifier("rf", max_depth=1 + hp.randint('max_depth', 15),
        #                               n_estimators=20 + hp.randint('n_estimators', 300))
        estim = HyperoptEstimator(classifier=rf, preprocessing=[standard_scaler("myZscore")], algo=tpe.suggest,
                                  loss_fn=fn_AUC, max_evals=10, n_jobs=8)

    # estim.fit(x_train, y_train, cv_shuffle=True) # n_folds=2,
    # print(estim.score(x_test, y_test))
    # print(estim.predict(x_test))  # return label
    # print(estim.best_model())

    estim.fit(X, y, cv_shuffle=True)
    mdl = estim.best_model()
    return mdl

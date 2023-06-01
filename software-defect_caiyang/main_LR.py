import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.metrics import roc_auc_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from tuneParameters import tuneclassifier


def datapreprocessing(data):
    # remove ND，REXP，LA,LD
    X = data[['ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
    # log
    X = np.log(X + 1.1)
    X = np.nan_to_num(X)
    X_fix = data[['fix']]
    X = np.concatenate((X, X_fix), axis=1)
    y= data[['bug']]
    # X   array to dataframe
    #   dataframeto array
    X = np.array(X, dtype="float64")
    y = np.array(y, dtype="float64")
    # # y is one-dimensional array
    y = np.ravel(y)
    # effort = tst_X[:, 7]  la+ld
    effort = X['la'] + X['ld']
    return X, y,effort

# 传统评估方法
def evaluate(y_true, y_pred):
    # pre<0.5  =0；  pre>=0.5  =1；
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2 * recall * precision / (recall + precision)
    Pf = fp / (fp + tn)
    AUC = roc_auc_score(y_true, y_pred)
    MCC = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
    MCC = (tp * tn - fn * fp) / np.sqrt(MCC)
    return recall, precision, F1, Pf, AUC, MCC


def evaluate_all(tst_pred, effort, tst_y):
    Popt = rankmeasure_e(tst_pred, effort, tst_y)
    Erecall, Eprecision, Efmeasure, PMI, IFA = rankmeasure_c(tst_pred, effort, tst_y)
    recall, precision, F1, Pf, AUC, MCC = evaluate(tst_y, tst_pred)
    return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision, F1, Pf, AUC, MCC

def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
    # tune classification parameters
    mdl = tuneclassifier(trn_X, trn_y, "LR")
    mdl = mdl['learner']
    mdl.fit(trn_X, trn_y)
    tst_pred = mdl.predict(tst_X)
    # modelLR = LogisticRegression(max_iter=2000)
    # modelLR.fit(trn_X, trn_y)
    # tst_pred = modelLR.predict_proba(tst_X)
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC = evaluate_all(tst_pred[:, 1], effort,tst_y)
    return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC

def save_results_to_csv(results, method, dataset):
    for key, value in results.items():
        df = pd.DataFrame(value)
        df.columns = ['Popt', 'Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
        outpath = f'./output/{method}-{dataset}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)

datasets=["nova","broadleaf","spring-integration","neutron","brackets","tomcat","fabric","jgroups","camel"]
method = "LR"
for j in range(10):
    # timewise
    dataset = datasets[j]
    fname = dataset + ".arff"
    file = "./datasets/" + fname
    data = pd.read_csv(file)
    # change format
    data["commitdate"] = pd.to_datetime(data["commitTime"]).dt.strftime('%Y-%m-%d')
    data = data.sort_values("commitdate")
    # type(bool) to 01
    data = data.fillna(0)
    for u in data.columns:
        # nan to 0,object to bool
        if data[u].dtype == object:
            data[u] = data[u].astype(bool)

        if data[u].dtype == bool:
            data[u] = data[u].astype('int')
    # ordered data by commitTime
    unimon = data['commitdate'].unique()
    unimon.sort()
    totalFolds = len(unimon)
    sub = [None] * totalFolds
    # timewise
    gap = 2
    # test
    totalFolds = 26
    for fold in range(totalFolds):
        sub[fold] = data[data['commitdate'] == unimon[fold]]
    for fold in range(totalFolds):
        if (fold + 6 > totalFolds):
            # index 0...
            continue
        trn = pd.concat([sub[fold], sub[fold + 1]])  # train set
        tst = pd.concat([sub[fold + 2 + gap], sub[fold + 3 + gap]])  # test set
        # (1) datapreprocessing
        trn_X, trn_y, Eeffort = datapreprocessing(trn)
        tst_X, tst_y, effort = datapreprocessing(tst)

    sampling_methods = {
        "none": None,
        "enn": EditedNearestNeighbours(random_state=0),
        "rum": RandomUnderSampler(random_state=0),
        "nm":  NearMiss(random_state=0),
        "tlr": TomekLinks(random_state=0),
        "rom": RandomOverSampler(random_state=0),
        "smo": SMOTE(random_state=0),
        "bsmote": BorderlineSMOTE(random_state=0),
        "csmote": SMOTETomek(random_state=0),
        "oss":  OneSidedSelection(random_state=0),
        "cenn":  SMOTEENN(random_state=0),
    }
    results = {key: np.zeros(shape=(50, 12)) for key in sampling_methods.keys()}
    for i in range(50):
        for method, sampler in sampling_methods.items():
            if sampler is None:
                n_X, n_y = trn_X, trn_y
            else:
                n_X, n_y = sampler.fit_resample(trn_X, trn_y)
            results[method][i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
            print(f"{method} is okay~")
    save_results_to_csv(results, method, dataset)
    print("running is okay~")


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#from statistics import harmonic_mean
#import struct
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from smo import smote
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

datasets=["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
method = "NB"
best_k=np.zeros(10)
for j in range(10):
    dataset=datasets[j]
    with open("./datasets/"+dataset+".arff", encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        data = pd.read_csv(f, header=None)
        data.columns = header
    data
    data = data.fillna(0)
    #布尔类型转01
    for u in data.columns:
        # nan转0,object强制转换bool
        if data[u].dtype == object:
            data[u] = data[u].astype(bool)

        if data[u].dtype == bool :
            data[u] = data[u].astype('int')
    X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
    y = data[['contains_bug']]
    # 划分数据集
    def divideddata(X, y):
        # 按照5：5划分训练集和数据集
        trn_Xq, tst_Xq, trn_y, tst_y = train_test_split(X, y, test_size=0.5)
        # 移除ND，REXP，LA,LD度量
        trn_X = trn_Xq[['ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        # log归一化处理
        trn_X = np.log(trn_X + 1.1)
        trn_X = np.nan_to_num(trn_X)
        trn_X_fix = trn_Xq[['fix']]
        trn_X = np.concatenate((trn_X, trn_X_fix), axis=1)

        tst_X = tst_Xq[['ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        # log归一化处理
        tst_X = np.log(tst_X + 1.1)
        tst_X = np.nan_to_num(tst_X)
        tst_X_fix = tst_Xq[['fix']]
        tst_X = np.concatenate((tst_X, tst_X_fix), axis=1)

        effort = tst_Xq['la'] + tst_Xq['ld']

        # 由dataframe转换成矩阵
        trn_X = np.array(trn_X, dtype="float64")
        tst_X = np.array(tst_X, dtype="float64")
        trn_y = np.array(trn_y, dtype="float64")
        tst_y = np.array(tst_y, dtype="float64")
        effort = np.array(effort, dtype="float64")
        # # y应该是一维数组格式
        trn_y = trn_y.reshape(-1)
        tst_y = tst_y.reshape(-1)
        return trn_X, tst_X, trn_y, tst_y, effort

    # non-effort
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
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        F1 = 2 * recall * precision / (recall + precision)
        Pf = fp / (fp + tn)
        G1 = 2 * recall * (1 - Pf) / (recall + (1 - Pf))
        AUC = roc_auc_score(y_true, y_pred)
        MCC = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
        MCC = (tp * tn - fn * fp) / np.sqrt(MCC)
        return recall, precision, F1, Pf, AUC, MCC

    def AUC_evaluate(y_true, y_pred):
        # pre<0.5  =0；  pre>=0.5  =1；
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        AUC = roc_auc_score(y_true, y_pred)
        return AUC

    def evaluate_all(tst_pred, effort, tst_y):
        ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, F1, Pf, AUC, MCC = evaluate(tst_y, tst_pred)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC

    # 逻辑回归分类器
    def nb_opt(X, y, class_weight=None):
        k_list = [1, 2, 3]
        result = np.zeros(shape=(3, 10))
        kf = KFold(n_splits=10, shuffle=True)
        for k in k_list:
           i = 0
           if k == 1:
             clf = GaussianNB()
           elif k == 2:
             clf = BernoulliNB()
           elif k == 3:
             clf = MultinomialNB()
           for train_index, test_index in kf.split(X, y):
                # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
                trn_x, trn_y = X[train_index], y[train_index]  # 本组训练集
                trn_x, trn_y = smote(trn_x, trn_y, balanced_ratio=1, k=5)
                tst_x, tst_y = X[test_index], y[test_index]  # 本组验证集
                trn_x = np.maximum(trn_x, -trn_x)
                clf.fit(trn_x, trn_y)
                tst_pred = clf.predict(tst_x)
                result[k-1,i]= AUC_evaluate(tst_y, tst_pred)
                i = i + 1
        return result

        # 高斯贝叶斯
    def nb_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = GaussianNB()
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC = evaluate_all(tst_pred, effort, tst_y)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC

        #伯努利贝叶斯
    def ber_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = BernoulliNB()
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC = evaluate_all(tst_pred, effort, tst_y)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC


    #多项式贝叶斯
    def mul_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = MultinomialNB()
        trn_X = np.maximum(trn_X, -trn_X)
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC = evaluate_all(tst_pred, effort, tst_y)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC

    results = np.zeros(shape=(3, 1))
    optnb= np.zeros(shape=(50,12))
    nb= np.zeros(shape=(50,12))
    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        result=nb_opt(trn_X, trn_y, class_weight=None)
        results=np.concatenate((results, result), axis=1)
    k_list = [1, 2, 3]
    bst_k = k_list[0]
    bst_m = 0
    for i in range(3):
        if np.mean(results[i,:]) > bst_m:
            bst_k = k_list[i]
            bst_m = np.mean(results[i,:])
    print("bst_k is:")
    print(bst_k)

    best_k[j] = bst_k

    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X, y)
        n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1, k=5)
        if bst_k == 1:
            optnb[i, :] = nb_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        elif bst_k == 2:
            optnb[i, :] = ber_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        elif bst_k == 3:
            optnb[i, :] = mul_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        nb[i, :] = nb_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("optnb is okay~")

    optnb = pd.DataFrame(optnb)
    optnb.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA','recall', 'precision',  'F1', 'Pf', 'AUC', 'MCC']
    optnboutpath = './output-opt/'+method+'-'+dataset+'-opt' + '.csv'
    optnb.to_csv(optnboutpath, index=True, header=True)

    nb = pd.DataFrame(nb)
    nb.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA','recall', 'precision',  'F1', 'Pf', 'AUC', 'MCC']
    nboutpath = './output-opt/' + method + '-' + dataset + '-smo' + '.csv'
    nb.to_csv(nboutpath, index=True, header=True)

best_k = pd.DataFrame(best_k)
best_k.index = ["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
best_k.columns = ["opt-para"]
best_koutpath = './output-opt/' + method +  '-best-k' + '.csv'
best_k.to_csv(best_koutpath, index=True, header=True)





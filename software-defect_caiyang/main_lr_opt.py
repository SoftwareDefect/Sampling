# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#from statistics import harmonic_mean
#import struct
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
from smo import smote
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

datasets=["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
method = "LR"
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
    #contains_bug列 object转int

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

    def AUC_evaluate(y_true, y_pred):
        # pre<0.5  =0；  pre>=0.5  =1；
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        AUC = roc_auc_score(y_true, y_pred)
        return AUC

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

    def evaluate_all(tst_pred, effort, tst_y):
        ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, F1, Pf, AUC, MCC = evaluate(tst_y, tst_pred)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC

    # 逻辑回归分类器
    def lr_opt(X, y, class_weight=None):
        k_list = [1, 2, 3, 4, 5]
        result = np.zeros(shape=(5, 10))
        kf = KFold(n_splits=10, shuffle=True)
        for k in k_list:
            i=0
            modelLR = LogisticRegression(max_iter=2000, C=k)
            for train_index, test_index in kf.split(X, y):
                # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
                trn_x, trn_y = X[train_index], y[train_index]  # 本组训练集
                trn_x, trn_y = smote(trn_x, trn_y, balanced_ratio=1, k=5)
                tst_x, tst_y = X[test_index], y[test_index]  # 本组验证集
                # 训练本组的数据，并计算准确率
                modelLR.fit(trn_x, trn_y)
                tst_pred = modelLR.predict(tst_x)
                result[k-1,i] = AUC_evaluate(tst_pred,tst_y)
                i=i+1
        return result


    # 逻辑回归分类器
    def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, k,class_weight=None):
        modelLR = LogisticRegression(max_iter=2000,C=k)
        modelLR.fit(trn_X, trn_y)
        tst_pred = modelLR.predict_proba(tst_X)
        ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC = evaluate_all(tst_pred[:, 1], effort, tst_y)
        return ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, recall, precision, F1, Pf, AUC, MCC

    results = np.zeros(shape=(5, 1))
    optlr= np.zeros(shape=(50, 12))
    lr = np.zeros(shape=(50, 12))
    trn_X, tst_X, trn_y, tst_y, effort = divideddata(X, y)
    for i in range(50):
        result=lr_opt(trn_X, trn_y, class_weight=None)
        results = np.concatenate((results, result), axis=1)
    k_list = [1, 2, 3, 4, 5]
    bst_k = k_list[0]
    bst_m = 0
    for k in range(5):
        if np.mean(results[k,:]) > bst_m:
            bst_k = k_list[k]
            bst_m = np.mean(results[k,:])
    print("bst_k is:")
    print(bst_k)
    best_k[j] = bst_k

    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X, y)
        n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1, k=5)
        optlr[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None,k=bst_k)
        lr[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None, k=1.0)
        print("optsmote is okay~")

    optlr = pd.DataFrame(optlr)
    optlr.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA','recall', 'precision',  'F1', 'Pf', 'AUC', 'MCC']
    optlroutpath = './output-opt-50/'+method+'-'+dataset+'-opt' + '.csv'
    optlr.to_csv(optlroutpath, index=True, header=True)

    lr = pd.DataFrame(lr)
    lr.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA','recall', 'precision',  'F1', 'Pf', 'AUC', 'MCC']
    lroutpath = './output-opt-50/' + method + '-' + dataset + '-smo' + '.csv'
    lr.to_csv(lroutpath, index=True, header=True)


best_k = pd.DataFrame(best_k)
best_k.index = ["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
best_k.columns = ["opt-para"]
best_koutpath = './output-opt-50/' + method +  '-best-k' + '.csv'
best_k.to_csv(best_koutpath, index=True, header=True)
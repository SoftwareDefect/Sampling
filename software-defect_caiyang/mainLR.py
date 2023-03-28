#采样后各个分类器的指标结果
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import OneSidedSelection

from tomekl_rm import tomek_link_rm
from cSmotet import combine_smote_tomek
from rus import random_undersampling
from nMiss import near_miss
from rom import random_oversampling
from smo import smote
from bSmote import borderline_smote
from cEnn import combine_enn
from ednn import edited_nn

datasets=["nova","broadleaf","spring-integration","neutron","brackets","tomcat","fabric","jgroups","camel"]
method = "LR"
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

        # effort = tst_X[:, 7]  la+ld
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
        return recall, precision,  F1, Pf, AUC,MCC

    def evaluate_all(tst_pred,effort,tst_y):
        Popt = rankmeasure_e(tst_pred, effort, tst_y)
        Erecall, Eprecision, Efmeasure, PMI, IFA = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision,  F1, Pf, AUC, MCC = evaluate(tst_y, tst_pred)
        return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC


    #逻辑回归
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        modelLR = LogisticRegression(max_iter=2000)
        modelLR.fit(trn_X, trn_y)
        tst_pred = modelLR.predict_proba(tst_X)
        Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC = evaluate_all(tst_pred[:, 1], effort,tst_y)
        return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC


    none = np.zeros(shape=(50, 12))
    rum = np.zeros(shape=(50, 12))
    nm = np.zeros(shape=(50, 12))
    enn = np.zeros(shape=(50, 12))
    tlr = np.zeros(shape=(50, 12))
    rom = np.zeros(shape=(50, 12))
    ooss = np.zeros(shape=(50, 12))
    smo = np.zeros(shape=(50, 12))
    bsmote = np.zeros(shape=(50, 12))
    csmote = np.zeros(shape=(50, 12))
    cenn = np.zeros(shape=(50, 12))
    eensemble = np.zeros(shape=(50, 12))
    bc= np.zeros(shape=(50, 12))
    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        none[i, :] = lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None)
        print("None is okay~")
        n_X, n_y = edited_nn(trn_X, trn_y, k=15)
        enn[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("edited_nn is okay~")
        n_X, n_y = random_undersampling(trn_X, trn_y, balanced_ratio=1)
        rum[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("random_undersampling is okay~")
        n_X, n_y = near_miss(trn_X, trn_y, k=3, balanced_ratio=1)
        nm[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("near_miss is okay~")
        n_X, n_y = tomek_link_rm(X=trn_X, y=trn_y)
        tlr[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("tomek_link_rm is okay~")
        n_X, n_y = random_oversampling(X=trn_X, y=trn_y, balanced_ratio=1)
        rom[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("random_oversampling is okay~")
        n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1,k=5)
        smo[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("smote is okay~")
        n_X, n_y = borderline_smote(trn_X, trn_y, balanced_ratio=1)
        bsmote[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("borderline_smote is okay~")
        n_X, n_y = combine_smote_tomek(trn_X, trn_y, balanced_ratio=1)
        csmote[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("combine_smote_tomek is okay~")
        oss = OneSidedSelection(random_state=42)
        n_X, n_y = oss.fit_resample(trn_X, trn_y)
        ooss[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("OneSidedSelection is okay~")
        n_X, n_y = combine_enn(trn_X, trn_y, k=7, balanced_ratio=1)
        cenn[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("combine_enn is okay~")

    none = pd.DataFrame(none)
    none.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    noneoutpath = './output/'+method+'-'+dataset+'-none.csv'
    none.to_csv(noneoutpath, index=True, header=True)


    rum = pd.DataFrame(rum)
    rum.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    rumoutpath = './output/'+method+'-'+dataset+'-rum' + '.csv'
    rum.to_csv(rumoutpath, index=True, header=True)

    nm = pd.DataFrame(nm)
    nm.columns = ['Popt','Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    nmoutpath ='./output/'+method+'-'+dataset+'-nm' + '.csv'
    nm.to_csv(nmoutpath, index=True, header=True)

    enn = pd.DataFrame(enn)
    enn.columns = ['Popt','Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    ennoutpath = './output/'+method+'-'+dataset+'-enn' + '.csv'
    enn.to_csv(ennoutpath, index=True, header=True)

    tlr = pd.DataFrame(tlr)
    tlr.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    tlroutpath = './output/'+method+'-'+dataset+'-tlr' + '.csv'
    tlr.to_csv(tlroutpath, index=True, header=True)

    rom = pd.DataFrame(rom)
    rom.columns = ['Popt','Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    romoutpath = './output/'+method+'-'+dataset+'-rom' + '.csv'
    rom.to_csv(romoutpath, index=True, header=True)

    smo = pd.DataFrame(smo)
    smo.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    smooutpath = './output/'+method+'-'+dataset+'-smo' + '.csv'
    smo.to_csv(smooutpath, index=True, header=True)

    bsmote = pd.DataFrame(bsmote)
    bsmote.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    bsmoteoutpath = './output/'+method+'-'+dataset+'-bsmote' + '.csv'
    bsmote.to_csv(bsmoteoutpath, index=True, header=True)

    csmote = pd.DataFrame(csmote)
    csmote.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    csmoteoutpath = './output/'+method+'-'+dataset+'-csmote' + '.csv'
    csmote.to_csv(csmoteoutpath, index=True, header=True)

    ooss = pd.DataFrame(ooss)
    ooss.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    oossoutpath = './output/' + method + '-' + dataset + '-oss' + '.csv'
    ooss.to_csv(oossoutpath, index=True, header=True)

    cenn = pd.DataFrame(cenn)
    cenn.columns = ['Popt', 'Erecall','Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    cennoutpath = './output/'+method+'-'+dataset+'-cenn' + '.csv'
    cenn.to_csv(cennoutpath, index=True, header=True)

    print("running is okay~")



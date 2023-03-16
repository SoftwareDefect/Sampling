# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from statistics import harmonic_mean
import struct
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

from EasyEn import EasyEnsemble
from tomekl_rm import tomek_link_rm
from cSmotet import combine_smote_tomek
from rus import random_undersampling
from nMiss import near_miss
from rom import random_oversampling
from smo import smote
from bSmote import borderline_smote
from cEnn import combine_enn
from EasyEn import EasyEnsemble
from bs import BalanceCascade
from cnn import condensed_nn
from ednn import edited_nn
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.metrics import roc_auc_score

datasets=["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
#datasets=["nova","npm"]
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
    # 由dataframe转换成矩阵
    # X = np.array(X)
    # y = np.array(y)
    # # y应该是一维数组格式
    # y = y.reshape(-1)
    #决策树 data数据集变偶数,重新赋X,y
    def odd_divided(data):
        if len(data) % 2 != 0:
            data = data.iloc[0:len(data) - 1]
            X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
            y = data[['contains_bug']]
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        return trn_X, tst_X, trn_y, tst_y, effort

    # 划分数据集
    def divideddata(X, y):
        # 按照7：3划分训练集和数据集
        trn_Xq, tst_Xq, trn_y, tst_y = train_test_split(X, y, test_size=0.5)
        # 移除ND，REXP，LA,LD度量
        trn_X = trn_Xq[['ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        # # zscore归一化处理
        # trn_X = preprocessing.scale(trn_X)
        # log归一化处理
        trn_X = np.log(trn_X + 1.1)
        trn_X = np.nan_to_num(trn_X)
        trn_X_fix = trn_Xq[['fix']]
        trn_X = np.concatenate((trn_X, trn_X_fix), axis=1)

        tst_X = tst_Xq[['ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        # # zscore归一化处理
        # tst_X = preprocessing.scale(tst_X)
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

        # log归一化
        # trn_X=np.log(trn_X+1)
        # trn_X = np.nan_to_num(trn_X)
        # tst_X=np.log(tst_X+1)
        # tst_X = np.nan_to_num(tst_X)

        return trn_X, tst_X, trn_y, tst_y, effort


    def evaluate_all(tst_pred,effort,tst_y):
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


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
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        F1 = 2 * recall * precision / (recall + precision)
        Pf = fp / (fp + tn)
        G1 = 2 * recall * (1 - Pf) / (recall + (1 - Pf))
        AUC = roc_auc_score(y_true, y_pred)
        MCC = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
        MCC = (tp * tn - fn * fp) / np.sqrt(MCC)
        return recall, precision, accuracy, F1, Pf, G1, AUC,MCC

    #逻辑回归分类器
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        modelLR = LogisticRegression(max_iter=2000)
        modelLR.fit(trn_X, trn_y)
        tst_pred = modelLR.predict_proba(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA,ePopt = rankmeasure_e( tst_pred[:, 1], effort,tst_y)
        cErecall, cEprecision, cEfmeasure,cPMI, cIFA,cPopt = rankmeasure_c(tst_pred[:, 1], effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC,MCC = evaluate(tst_y, tst_pred[:, 1])
        return Erecall, Eprecision, Efmeasure,  ePMI, eIFA,ePopt,cErecall, cEprecision, cEfmeasure, cPMI, cIFA,cPopt, recall, precision, accuracy, F1, Pf, G1, AUC,MCC
    #决策树
    #trn_X 训练数据，trn_y 训练标签，tst_X 测试数据，tst_y测试标签
    def dt_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):

        vec = DictVectorizer(sparse=False)
        trn_X =  pd.DataFrame(trn_X)
        trn_y =  pd.DataFrame(trn_y)
        tst_X =  pd.DataFrame(tst_X)
        X_train = vec.fit_transform(trn_X.to_dict(orient='record'))
        Y_train = vec.fit_transform(trn_y.to_dict(orient='record'))
        clf = tree.DecisionTreeClassifier(criterion='gini')
        model = clf.fit(X_train, Y_train.astype('float'))
        tst_pred = model.predict(tst_X)

        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e( tst_pred, effort,tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt= rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC,MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC

    #随机森林
    def rf_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        sc = StandardScaler()
        trn_X = sc.fit_transform(trn_X)
        tst_X = sc.transform(tst_X)
        # 训练随机森林解决回归问题
        clf = RandomForestClassifier(n_estimators=200, random_state=1)
        #n_estimators表示数的个数，数值越大越好，但是会占用内存多
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


    # 高斯贝叶斯
    def nb_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = GaussianNB()
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


    none = np.zeros(shape=(50, 20))
    # rum = np.zeros(shape=(50, 20))
    # nm = np.zeros(shape=(50, 20))
    # enn = np.zeros(shape=(50, 20))
    # tlr = np.zeros(shape=(50, 20))
    # rom = np.zeros(shape=(50, 20))
    #cnn = np.zeros(shape=(50, 20))
    # smo = np.zeros(shape=(50, 20))
    # bsmote = np.zeros(shape=(50, 20))
    # csmote = np.zeros(shape=(50, 20))
    # cenn = np.zeros(shape=(50, 20))
    # eensemble = np.zeros(shape=(50, 20))
    # bc= np.zeros(shape=(50, 20))
    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        none[i, :] = rf_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None)
        print("None is okay~")
        # n_X, n_y = random_undersampling(trn_X, trn_y, balanced_ratio=1)
        # rum[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("random_undersampling is okay~")
        # n_X, n_y = near_miss(trn_X, trn_y, k=3, balanced_ratio=1)
        # nm[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("near_miss is okay~")
        # n_X, n_y = edited_nn(trn_X, trn_y, k=15)
        # enn[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("edited_nn is okay~")
        # n_X, n_y = tomek_link_rm(X=trn_X, y=trn_y)
        # tlr[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("tomek_link_rm is okay~")
        # n_X, n_y = random_oversampling(X=trn_X, y=trn_y, balanced_ratio=1)
        # rom[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("random_oversampling is okay~")
        # n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1,k=5)
        # smo[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("smote is okay~")
        # n_X, n_y = borderline_smote(trn_X, trn_y, balanced_ratio=1)
        # bsmote[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("borderline_smote is okay~")
        # n_X, n_y = combine_smote_tomek(trn_X, trn_y, balanced_ratio=1)
        # csmote[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("combine_smote_tomek is okay~")
        # n_X, n_y = condensed_nn(trn_X, trn_y)
        # cnn[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("condensed_nn is okay~")
        # n_X, n_y = combine_enn(trn_X, trn_y, k=7, balanced_ratio=1)
        # cenn[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        # print("combine_enn is okay~")

    none = pd.DataFrame(none)
    none.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    noneoutpath = './output/'+method+'-'+dataset+'-none.csv'
    none.to_csv(noneoutpath, index=True, header=True)
    #
    # rum = pd.DataFrame(rum)
    # rum.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # rumoutpath = './output-zscore/'+method+'-'+dataset+'-rum' + '.csv'
    # rum.to_csv(rumoutpath, index=True, header=True)
    #
    # nm = pd.DataFrame(nm)
    # nm.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # nmoutpath ='./output-zscore/'+method+'-'+dataset+'-nm' + '.csv'
    # nm.to_csv(nmoutpath, index=True, header=True)
    #
    # enn = pd.DataFrame(enn)
    # enn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # ennoutpath = './output-zscore/'+method+'-'+dataset+'-enn' + '.csv'
    # enn.to_csv(ennoutpath, index=True, header=True)
    #
    # tlr = pd.DataFrame(tlr)
    # tlr.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # tlroutpath = './output-zscore/'+method+'-'+dataset+'-tlr' + '.csv'
    # tlr.to_csv(tlroutpath, index=True, header=True)
    #
    # rom = pd.DataFrame(rom)
    # rom.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # romoutpath = './output-zscore/'+method+'-'+dataset+'-rom' + '.csv'
    # rom.to_csv(romoutpath, index=True, header=True)
    #
    # smo = pd.DataFrame(smo)
    # smo.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # smooutpath = './output-zscore/'+method+'-'+dataset+'-smo' + '.csv'
    # smo.to_csv(smooutpath, index=True, header=True)
    #
    # bsmote = pd.DataFrame(bsmote)
    # bsmote.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # bsmoteoutpath = './output-zscore/'+method+'-'+dataset+'-bsmote' + '.csv'
    # bsmote.to_csv(bsmoteoutpath, index=True, header=True)
    #
    # csmote = pd.DataFrame(csmote)
    # csmote.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # csmoteoutpath = './output-zscore/'+method+'-'+dataset+'-csmote' + '.csv'
    # csmote.to_csv(csmoteoutpath, index=True, header=True)

    cnn = pd.DataFrame(cnn)
    cnn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    cnnoutpath = './output/'+method+'-'+dataset+'-cnn' + '.csv'
    cnn.to_csv(cnnoutpath, index=True, header=True)

    # cenn = pd.DataFrame(cenn)
    # cenn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    # cennoutpath = './output-zscore/'+method+'-'+dataset+'-cenn' + '.csv'
    # cenn.to_csv(cennoutpath, index=True, header=True)

    print("running is okay~")

    # measure.columns = ['recall', 'precision','accuracy','F1','G1']
    # 看csv的文件
    # import pandas as pd
    # data = pd.read_csv(r'D:/software_defect-caiyang/LRmeasure2.csv',sep=',',header='infer')
    # # #看npy的文件
    # import numpy as np
    # cumXs = np.load(file="D:/software_defect-yaner/measure.npy",allow_pickle=True)




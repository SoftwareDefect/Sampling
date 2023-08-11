import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from rankMeasure_c import rank_measure


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

def datapreprocessing_all(trn,tst):
    # train label
    trn_y = trn['bug'].to_frame()
    # train data normalization
    selected_columns = ['ns', 'nd', 'nf', 'entrophy', 'la','ld', 'lt', 'fix', 'age', 'ndev', 'nuc', 'exp', 'rexp', 'sexp']
    trn_X = trn.loc[:, selected_columns]
    trn_X = np.log(trn_X + 1.1)
    trn_X = trn_X.replace([np.nan, np.inf, -np.inf], 0)

    # tst label
    tst_y = tst['bug'].to_frame()
    #tst data
    selected_columns = ['ns', 'nd', 'nf', 'entrophy', 'la','ld', 'lt', 'fix', 'age', 'ndev', 'nuc', 'exp', 'rexp', 'sexp']
    tst_X = tst.loc[:, selected_columns]
    # test data normalization
    tst_X = np.log(tst_X + 1.1)
    tst_X = tst_X.replace([np.nan, np.inf, -np.inf], 0)
    # tst effort
    effort = (tst['la'] + tst['ld']).to_frame()
    #effort = decChurn(tst)
    effort.columns = ['effort']


    # dataframeto array
    trn_X = np.array(trn_X, dtype="float64")
    trn_y = np.array(trn_y, dtype="float64")
    tst_X = np.array(tst_X, dtype="float64")
    tst_y = np.array(tst_y, dtype="float64")
    # # y is one-dimensional array
    trn_y = np.ravel(trn_y)
    tst_y = np.ravel(tst_y)
    effort = np.ravel(effort)
    return trn_X, trn_y, tst_X, tst_y, effort

def datapreprocessing(trn,tst,metric):
    # train label
    trn_y = trn['bug'].to_frame()
    # train data normalization
    trn_X = trn[metric].to_frame()
    trn_X = np.log(trn_X + 1.1)
    trn_X = trn_X.replace([np.nan, np.inf, -np.inf], 0)

    # tst label
    tst_y = tst['bug'].to_frame()
    #tst data
    tst_X = tst[metric].to_frame()
    # test data normalization
    tst_X = np.log(tst_X + 1.1)
    tst_X = tst_X.replace([np.nan, np.inf, -np.inf], 0)
    # tst effort
    effort = (tst['la'] + tst['ld']).to_frame()
    #effort = decChurn(tst)
    effort.columns = ['effort']


    # dataframeto array
    trn_X = np.array(trn_X, dtype="float64")
    trn_y = np.array(trn_y, dtype="float64")
    tst_X = np.array(tst_X, dtype="float64")
    tst_y = np.array(tst_y, dtype="float64")
    # # y is one-dimensional array
    trn_y = np.ravel(trn_y)
    tst_y = np.ravel(tst_y)
    effort = np.ravel(effort)
    return trn_X, trn_y, tst_X, tst_y, effort

def unimon_data(data):
    # replace NaN and infinite values with 0
    data = data.replace([np.nan, np.inf, -np.inf], 0)
    # bool to 01
    data = data.fillna(0)
    # change format
    data["commitdate"] = pd.to_datetime(data["commitTime"]).dt.strftime('%Y-%m')
    data = data.sort_values("commitdate")
    # ordered data by commitTime
    unimon = data['commitdate'].unique()
    unimon.sort()
    totalFolds = len(unimon)
    sub = [None] * totalFolds
    for fold in range(totalFolds):
        sub[fold] = data[data['commitdate'] == unimon[fold]]
    return totalFolds, sub
# traditional evaluate
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
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rank_measure(tst_pred, effort, tst_y)
    recall, precision, F1, Pf, AUC, MCC = evaluate(tst_y, tst_pred)
    return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision, F1, Pf, AUC, MCC


def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
    modelLR = LogisticRegression(max_iter=2000)
    modelLR.fit(trn_X, trn_y)
    tst_pred = modelLR.predict_proba(tst_X)
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC = evaluate_all(tst_pred[:, 1],effort, tst_y)
    return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC

def save_results_to_csv(results, dataset,metric):

    df = pd.DataFrame(results)
    # Drop rows with NaN values
    df = df.dropna()
    # Reset the index
    df = df.reset_index(drop=True)
    df.columns = ['Popt', 'Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
    outpath = f'./output/output-metrics/{metric}-{dataset}.csv'
    df.to_csv(outpath, index=True, header=True)


#datasets=["spring-integration","broadleaf","npm","nova","neutron","brackets","tomcat","fabric","jgroups","camel"]


#if __name__ == '__main__':
def main(DATASETS):
    # sampling_methods
    sampling_methods = {
        "none": None,
    }
    data_metrics = ['all','ns','nd','nf','entrophy','la','ld','lt','fix','age','ndev','nuc','exp','rexp','sexp']
    for metric in data_metrics:
        for j in range(10):
            # read data
            dataset = DATASETS[j]
            fname = dataset + ".csv"
            file = "D:/Git/Sampling/software-defect_caiyang/datasets/" + fname
            data = pd.read_csv(file)
            # timewise
            gap = 2
            #divide the data of same month into same fold
            totalFolds, sub = unimon_data(data)
            results = np.zeros(shape=(0, 12))
            for fold in range(totalFolds):
                if (fold + 6 > totalFolds):
                    continue
                trn = pd.concat([sub[fold], sub[fold + 1]])  # train set
                tst = pd.concat([sub[fold + 2 + gap], sub[fold + 3 + gap], ])  # test set
                #datapreprocessing
                if(metric=='all'):
                    trn_X, trn_y, tst_X, tst_y, effort = datapreprocessing_all(trn, tst)
                else:
                    trn_X, trn_y, tst_X, tst_y, effort = datapreprocessing(trn,tst,metric)
                for method, sampler in sampling_methods.items():
                    if sampler is None:
                        n_X, n_y = trn_X, trn_y
                    else:
                        n_X, n_y = sampler.fit_resample(trn_X, trn_y)
                    # ensure test data is not single class
                    if list(n_y).count(1) < 2 or list(n_y).count(0) < 2 or list(tst_y).count(0) < 2 or list(tst_y).count(1) < 2:
                        break
                    result = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
                    results = np.vstack((results, result))
                    print(f"{method} is okay~")
            save_results_to_csv(results, dataset,metric)
            print("running is okay~")
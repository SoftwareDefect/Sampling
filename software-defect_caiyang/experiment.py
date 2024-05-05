import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from rankMeasure_c import rank_measure
from tunePara import tuneparameters


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


def datapreprocessing(trn,tst,tgap):
    # train label
    trn_y = trn['bug'].to_frame()
    # train data normalization
    trn_X = trn['la'].to_frame()
    trn_X = np.log(trn_X + 1)
    trn_X = trn_X.replace([np.nan, np.inf, -np.inf], 0)

    # tgap label
    tgap_y = tgap['bug'].to_frame()
    # tgap data normalization
    tgap_X = tgap['la'].to_frame()
    tgap_X = np.log(tgap_X + 1)
    tgap_X = tgap_X.replace([np.nan, np.inf, -np.inf], 0)

    # tst label
    tst_y = tst['bug'].to_frame()
    #tst data
    tst_X = tst['la'].to_frame()
    # test data normalization
    tst_X = np.log(tst_X + 1)
    tst_X = tst_X.replace([np.nan, np.inf, -np.inf], 0)
    # tst effort
    effort = (tst['la'] + tst['ld']).to_frame()
    effort.columns = ['effort']


    # dataframeto array
    trn_X = np.array(trn_X, dtype="float64")
    trn_y = np.array(trn_y, dtype="float64")
    tst_X = np.array(tst_X, dtype="float64")
    tst_y = np.array(tst_y, dtype="float64")
    tgap_X = np.array(tgap_X, dtype="float64")
    tgap_y = np.array(tgap_y, dtype="float64")
    # # y is one-dimensional array
    trn_y = np.ravel(trn_y)
    tst_y = np.ravel(tst_y)
    tgap_y = np.ravel(tgap_y)
    effort = np.ravel(effort)
    return trn_X, trn_y, tst_X, tst_y, tgap_X, tgap_y, effort


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

def lr_opt_predict(trn_X, trn_y, tgap_X, tgap_y , tst_X, tst_y, effort, class_weight=None):
    # tune classification parameters
    mdl = tuneparameters(trn_X, trn_y, tgap_X, tgap_y)
    #train model
    mdl.fit(trn_X, trn_y)
    #predict
    tst_pred = mdl.predict_proba(tst_X)
    #calculate performance
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC = evaluate_all(tst_pred[:, 1],effort, tst_y)
    return Popt, Erecall, Eprecision, Efmeasure, PMI, IFA, recall, precision,  F1, Pf, AUC, MCC

def save_results_to_csv(results, dataset,timeperiod,model_para):
    for key, value in results.items():
        df = pd.DataFrame(value)
        # Drop rows with NaN values
        df = df.dropna()
        # Reset the index
        df = df.reset_index(drop=True)
        df.columns = ['Popt', 'Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA', 'recall', 'precision', 'F1', 'Pf', 'AUC', 'MCC']
        outpath = f'./output/output-{timeperiod}months-{model_para}/{dataset}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)

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
    return totalFolds,sub

#if __name__ == '__main__':
def main(DATASETS,timeperiod,model_para):
    # sampling_methods
    sampling_methods = {
        "none": None,
        "enn": EditedNearestNeighbours(),
        "rum": RandomUnderSampler(),
        "nm": NearMiss(),
        "tlr": TomekLinks(),
        "rom": RandomOverSampler(),
        "smo": SMOTE(k_neighbors=1),
        "bsmote": BorderlineSMOTE(),
        "csmote": SMOTETomek(),
        "oss": OneSidedSelection(),
        "cenn": SMOTEENN(),
    }
    for j in range(10):
        # read data
        dataset = DATASETS[j]
        fname = dataset + ".csv"
        file = "D:/Git/Sampling/software-defect_caiyang/datasets/" + fname
        data = pd.read_csv(file)
        #timewise
        gap = timeperiod
        # divide the data of same month into same fold
        totalFolds, sub = unimon_data(data)
        results = {key: np.zeros(shape=(0, 12)) for key in sampling_methods.keys()}
        for fold in range(totalFolds):
            if (fold + gap * 3 > totalFolds):
                continue
            trn = pd.concat([sub[fold + i] for i in range(gap)])  # train set
            tgap = pd.concat([sub[fold + gap + i] for i in range(gap)])  # test set
            tst = pd.concat([sub[fold + gap * 2 + i] for i in range(gap)])  # test set
            # datapreprocessing
            trn_X, trn_y, tst_X, tst_y, tgap_X, tgap_y, effort = datapreprocessing(trn, tst, tgap)
            # ensure train data  : the number of defect > non defect
            if list(trn_y).count(1) < 6 or list(trn_y).count(1) > list(trn_y).count(0)  :
                continue
            if model_para == 'optimized':
                if  list(tgap_y).count(1) < 6 or list(tgap_y).count(1) > list(tgap_y).count(0) :
                    continue
            # ensure test data is not single class
            if list(tst_y).count(1) < 2 or list(tst_y).count(0) < 2:
                continue
            for method, sampler in sampling_methods.items():
                if sampler is None:
                    n_X, n_y = trn_X, trn_y
                    if model_para=='optimized':
                        n_gap_X, n_gap_y = tgap_X, tgap_y
                else:
                    n_X, n_y = sampler.fit_resample(trn_X, trn_y)
                    if model_para == 'optimized':
                        n_gap_X, n_gap_y = sampler.fit_resample(tgap_X, tgap_y)
                if model_para=='default':
                    # ensure test data is not single class
                    if list(n_y).count(1) < 2 or list(n_y).count(0) < 2:
                        break
                    result = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
                elif model_para=='optimized':
                    # ensure n_y and n_ga_y and tst_y is not single class
                    if list(n_y).count(1) < 2 or list(n_y).count(0) < 2 or list(n_gap_y).count(1) < 2 or list(
                            n_gap_y).count(0) < 2:
                        break
                    result = lr_opt_predict(n_X, n_y, n_gap_X, n_gap_y, tst_X, tst_y, effort, class_weight=None)
                else:
                    print("model_para is erroe")
                results[method] = np.vstack((results[method], result))
                print(f"{method} is okay~")
        save_results_to_csv(results, dataset,timeperiod,model_para)
        print("running is okay~")
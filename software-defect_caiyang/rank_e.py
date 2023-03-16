import numpy as np
import snoop

class Xy:
    def __init__(self):
        self.x = float()
        self.y = float()

class Graph:
    def __init__(self):
        self.pred = Xy()
        self.opt = Xy()
        self.wst = Xy()

graph=Graph()
xy=Xy()

def rankmeasure_e(predict_label, effort, test_label):
    #EALR
    #测试数据的个数length
    #python下标从0开始
    length = len(test_label) - 1
    effort = effort +1
    preDD = predict_label/effort
    density = test_label/ effort
    #effort = np.round(effort)
    #合并测试标签，预测标签和effort
    data=np.zeros(shape=(len(test_label), 3))
    data[:,0] = preDD
    data[:,1] = effort
    data[:,2] = test_label
    # 按照第一列降序，第二列升=序进行排列
    # data= sorted(data, key=lambda x: ( -x[0],x[1]))
    #按照第一列降序，第二列降序排列
    data = sorted(data, key=lambda x: (-x[0], x[1]))
    data = np.array(data)
    pred, graph.pred = computeArea(data, length)

    #actural defect density, 'optimal model'
    data=np.zeros(shape=(len(test_label), 3))
    data[:,0] = density
    data[:,1] = effort
    data[:,2] = test_label
    data = sorted(data, key=lambda x: (-x[0], x[1]))
    opt,graph.opt= computeArea(data, length)

    #worst model
    data=np.zeros(shape=(len(test_label), 3))
    data[:,0] = density
    data[:,1] = effort
    data[:,2] = test_label
    data = sorted(data, key=lambda x: (x[0], -x[1]))
    wst,graph.wst= computeArea(data, length)

    if opt-wst!=0 :
        Popt= (pred-wst)/(opt-wst)
    else :
        Popt=0.5

    return Popt


def computeArea(data, length):
    # python 下标从0开始
    data = np.array(data)
    cumXs = np.cumsum(data[:, 1]);
    cumYs = np.cumsum(data[:, 2]);

    Xs = cumXs / cumXs[length];
    Ys = cumYs / cumYs[length];

    xy.x = Xs;
    xy.y = Ys;
    area = np.trapz(xy.x, xy.y);
    return area,xy

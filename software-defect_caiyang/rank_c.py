import numpy as np
import snoop

#定义结构体
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

def rankmeasure_c(predict_label, effort, test_label):

    #CBS
    # 测试数据的个数length
    # python下标从0开始
    length = len(test_label) - 1
    for i in range(len(predict_label)):
        if predict_label[i] < 0.5:
            predict_label[i] = 0
        else:
            predict_label[i] = 1
    data=np.zeros(shape=(len(test_label), 3))
    effort = np.round(effort)
    density = test_label / effort

    # 按照第一列降序，第二列升=序进行排列
    data[:,0] = predict_label
    data[:,1] = effort
    data[:,2] = test_label
    data= sorted(data, key=lambda x: ( -x[0],x[1]))
    data = np.array(data)
    pred, graph.pred = computeArea(data, length)
    cErecall, cEprecision, cEfmeasure, cPMI, cIFA  = computeMeasure(data, length)

    # actural defect density, 'optimal model'
    data = np.zeros(shape=(len(test_label), 3))
    data[:, 0] = density
    data[:, 1] = effort
    data[:, 2] = test_label
    data = sorted(data, key=lambda x: (-x[0], x[1]))
    opt, graph.opt = computeArea(data, length)

    # worst model
    data = np.zeros(shape=(len(test_label), 3))
    data[:, 0] = density
    data[:, 1] = effort
    data[:, 2] = test_label
    data = sorted(data, key=lambda x: (x[0], -x[1]))
    wst, graph.wst = computeArea(data, length)

    if opt - wst != 0:
        Popt = (pred - wst) / (opt - wst)
    else:
        Popt = 0.5

    return cErecall, cEprecision, cEfmeasure, cPMI, cIFA



def computeMeasure(data,length):
    cumXs = np.cumsum(data[:, 1])
    cumYs = np.cumsum(data[:, 2])
    Xs = cumXs / cumXs[length]
    #Xs是多维数组类型
    idx = next(iter(np.where(Xs>=0.2)[0]), -1)
    #pos=idx
    pos= idx+1

    Erecall = cumYs[idx]/ cumYs[length]

    Eprecision = cumYs[idx]/ pos



    if Erecall + Eprecision!=0 :
       Efmeasure = 2 * Erecall * Eprecision / (Erecall + Eprecision)

    else:
          Efmeasure = 0

    PMI = pos / length

    Iidx = next(iter(np.where(cumYs==1)[0]), -1)
    IFA = Iidx+1
    return Erecall,Eprecision,Efmeasure,PMI,IFA

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
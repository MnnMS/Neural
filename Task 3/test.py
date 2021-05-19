import numpy as np
from back_propagation import *
from sklearn.metrics import confusion_matrix

def test(X,Y,weights,layers,activ_fun,netVal,bias):
    accuracy = 0
    L = X.shape[0]
    y_true = []
    y_pred = []
    index = len(layers) - 1
    for i in range(0,L):
        _,neurons = forwardProp(X[i],weights,layers,activ_fun,netVal,bias)
        yHatoutput = neurons[index]
        modOutput , mx_ind = mx(yHatoutput)
        y = Y[i]
        y_true.append(clas(y))
        y_pred.append(mx_ind)

    accuracy = ((sum(np.diagonal(confusion_matrix(y_true, y_pred))))/L) * 100
    return confusion_matrix(y_true, y_pred).astype(int),accuracy

def mx(list1):
    output = [0, 0, 0]
    mx = max(list1)
    list1 = list(list1)
    mx_index = list1.index(mx)
    output[mx_index] = 1
    return output,mx_index
def clas(l):
    if l[0] == 1:
        return 0
    elif l[1] == 1:
        return 1
    elif l[2] == 1:
        return 2

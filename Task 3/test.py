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
        mx_ind = mx(yHatoutput)
        y = Y[i]
        y_true.append(y)
        y_pred.append(mx_ind)

    #matrix = test2(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    accuracy = ((sum(np.diagonal(matrix)))/L) * 100
    return matrix.astype(int),accuracy

def mx(list1):
    mx = max(list1)
    list1 = list(list1)
    mx_index = list1.index(mx)
    return mx_index
def clas(l):
    if l[0] == 1:
        return 0
    elif l[1] == 1:
        return 1
    elif l[2] == 1:
        return 2
def test2 (y_act, y_pred ):
    classes = np.unique(y_act)
    confusion_matrix = np.zeros([3, 3])
    for i in range(len(y_act)):
        index = np.where(classes == y_act[i])
        if y_pred[i] == y_act[i]:
             confusion_matrix[index, index] += 1
        else:
             index_y = np.where(classes == y_pred[i])
             confusion_matrix[index, index_y] += 1
    return confusion_matrix
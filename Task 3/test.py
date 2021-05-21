from back_propagation import *
from sklearn.metrics import confusion_matrix

def test(X,Y,weights,layers,activ_fun,netVal,bias,bonus_val):
    L = X.shape[0]
    y_true = []
    y_pred = []
    index = len(layers) - 1

    for i in range(0,L):
        _,neurons = forwardProp(X[i],weights,layers,activ_fun,netVal,bias)
        yHatoutput = neurons[index]
        mx_ind = mx(yHatoutput)
        y = Y[i] if bonus_val else clas(Y[i])
        y_true.append(y)
        y_pred.append(mx_ind)

    matrix = confusion_matrix(y_true, y_pred)
    accuracy = ((sum(np.diagonal(matrix)))/L) * 100
    return matrix.astype(int),accuracy

def mx(list1):
    mx = max(list1)
    list1 = list(list1)
    mx_index = list1.index(mx)
    return mx_index

def clas(l):
    list1 = list(l)
    return list1.index(1)

import numpy as np
from back_propagation import *
from sklearn.metrics import confusion_matrix

def test(X,Y,weights,layers,activ_fun,netVal,bias):
    accuracy = 0
    L = X.shape[0]
    y_true = []
    y_pred = []
    for i in range(0,L):
        _,neurons = forwardProp(X[i],weights,layers,activ_fun,netVal,bias)
        yHatoutput = neurons.pop()
        modOutput = mx(yHatoutput)
        if modOutput == Y[i]:
               accuracy += 1

    #confusion_matrix(y_true, y_pred)
    # accuracy = ((sum(np.diagonal(confusion_matrix)))/total) * 100
    return (accuracy/60)
#confusion_matrix.astype(int)
def mx(list1):
    output = [0, 0, 0]
    mx = max(list1)
    list1 = list(list1)
    mx_index = list1.index(mx)
    output[mx_index] = 1
    return output

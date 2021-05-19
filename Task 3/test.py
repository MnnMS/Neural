import numpy as np
from back_propagation import *



# def test(X,Y,weights,layers,activ_fun):
#     # classes = np.unique(T)
#     # num_of_classes = classes.size
#     # confusion_matrix = np.zeros([num_of_classes, num_of_classes])
#     # outPutClasses = 3
#     # layers = list(np.asarray(layers) + bias)
#     # layers.insert(0, X.shape[1])
#     # layers.append(outPutClasses)
#     # activation = []
#     # netVal = []
#     # if activ_fun == 0:
#     #     activation.append(lambda: sigmoid(*netVal))
#     # else:
#     #     activation.append(lambda: tangetHyperbolic(*netVal))
#     # L = X.shape[0]
#     # for i in range(0,L):
#     #     weights,neurons = forwardProp(X[i],Y[i],weights,layers,activation,*netVal)
#     #     yHatoutput = neurons.pop()
#     #     mx_index = mx(yHatoutput)
#     #     y = Y[i]
#     #     index = np.where(classes == Y[i])
#     #     if yHatoutput[mx_index] == y[mx_index]:
#     #         confusion_matrix[index, index] += 1
#     #
#     # accuracy = ((sum(np.diagonal(confusion_matrix)))/total) * 100
#     # return confusion_matrix.astype(int), accuracy

def test(X,Y,weights,layers,activ_fun,netVal,bias):
    accuracy = 0
    for i in range(60):
        _,output = forwardProp(X[i],weights,layers,activ_fun,netVal,bias)
        modifiedOutput = mx(output[-1])
        if modifiedOutput == Y[i]:
            accuracy += 1
    return (accuracy/60)
def mx(list1):
    # mx = max(list)
    # mx_index = yHatoutput.index(mx)
    # return mx_index
    output = [0,0,0]
    maxVal = max(list1)
    list1 = list(list1)
    maxIndex = list1.index(maxVal)
    output[maxIndex] = 1
    return output

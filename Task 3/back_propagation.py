import numpy as np


def tangetHyperbolic(netVal):

    return netVal

def sigmoid(netVal):
    output = 1/(1 + np.exp(-netVal))
    return output
#X,Y,layers,epochs,learn_rate,bias,activ_fun
layers = [2,2,2]
def forwardProp(X,Y,layers,epochs,learn_rate,bias,activ_fun):
    #adding activation functions to a list to choose between them easily
    activation = []
    netVal = [0]
    activation.append(lambda : sigmoid(*netVal))
    activation.append(lambda: tangetHyperbolic(*netVal))

    #initializing weights based on layers and neurons
    layers.insert(0,X.shape[1])
    layers.append(3)
    weights = []
    numOfWeights = 5 * X.shape[1]
    #numOfWeights = 5 * 5
    for i in range(len(layers)-1):
        numOfWeights = layers[i] * layers[i+1]
        weights.append(np.random.rand(1,numOfWeights)[0])
    if not bias:
        weights[0][0] = 0

    return weights

w = forwardProp(layers)
#print(np.dot(w[0]))
print(w)
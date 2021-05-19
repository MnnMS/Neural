import numpy as np
from back_propagation import *


def train(X,Y,layers,epochs,learn_rate,bias,activ_fun):
    # adding activation functions to a list to choose between them easily
    activation = []
    netVal = []
    if activ_fun == 0:
        activation.append(lambda: sigmoid(*netVal[0]))
    else:
        activation.append(lambda: tangetHyperbolic(*netVal[0]))

    # initializing weights based on layers and neurons
    outPutClasses = 3
    layers = list(np.asarray(layers))
    layers.insert(0, X.shape[1])
    layers.append(outPutClasses)
    weights = []
    for i in range(len(layers) - 1):
        if  i == 0:
            row = layers[i]
        else:
            row = layers[i] + bias

        col = layers[i+1]
        #numOfWeights = layers[i] * layers[i + 1]
        weights.append(np.random.rand(row , col))

    noOfSamples,noOfFeatures = X.shape
    while epochs:
        for i in range(noOfSamples):
            weights,neurons = forwardProp(X[i],weights,layers,activation,netVal,bias)
            errorSignal =backword(weights,neurons,Y[i],activ_fun,bias)
            weights = updateWeights(errorSignal,weights,learn_rate,neurons)
        epochs -= 1


    return weights ,neurons, layers , activation, netVal
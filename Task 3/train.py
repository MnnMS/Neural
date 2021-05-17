import numpy as np
from back_propagation import *


def train(X,Y,layers,epochs,learn_rate,bias,activ_fun):
    # adding activation functions to a list to choose between them easily
    activation = []
    netVal = []
    if activ_fun == 0:
        activation.append(lambda: sigmoid(*netVal))
    else:
        activation.append(lambda: tangetHyperbolic(*netVal))

    # initializing weights based on layers and neurons
    outPutClasses = 3
    layers.insert(0, X.shape[1])
    layers = list(np.asarray(layers) + bias)
    layers.append(outPutClasses)
    weights = []
    for i in range(len(layers) - 1):
        numOfWeights = layers[i] * layers[i + 1]
        weights.append(np.random.rand(1, numOfWeights)[0])

    noOfSamples,noOfFeatures = X.shape
    while epochs:
        for i in range(noOfSamples):
            weights,neurons = forwardProp(X[i],Y[i],weights,layers,activation,*netVal)


    return weights
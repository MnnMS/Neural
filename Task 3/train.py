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
    outPutClasses = len(np.unique(Y))
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
        weights.append(np.random.uniform(low=-1, high=1, size=(row , col)))
    # w1 = np.array([[0.16854289, 0.16357133], [0.55394706, 0.89541235], [-0.12771581, -0.35923104], [0.44864934, 0.04506766]])
    # w2 = np.array([[-0.86795458, 0.61664978, 0.47136797], [-0.49398027, 0.18329044, -0.60232547]])
    # w3 = np.array([[0.08666574, -0.27154152, -0.71973694], [-0.38224856, -0.83602736, -0.30448562],
    #       [-0.78799068, 0.60028769, -0.7052426]])
    # weights.append(w1)
    # weights.append(w2)
    # weights.append(w3)

    noOfSamples,noOfFeatures = X.shape
    while epochs:
        for i in range(noOfSamples):
            weights,neurons = forwardProp(X[i],weights,layers,activation,netVal,bias)
            errorSignal =backword(weights,neurons,Y[i],activ_fun,bias)
            weights = updateWeights(errorSignal,weights,learn_rate,neurons)
        epochs -= 1


    return weights ,neurons, layers , activation, netVal
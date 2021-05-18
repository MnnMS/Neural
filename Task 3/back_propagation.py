import numpy as np


def tangetHyperbolic(netVal):
    output = (1-np.exp(-netVal))/(1+np.exp(-netVal))
    return output

def sigmoid(netVal):
    output = 1/(1 + np.exp(-netVal))
    return output

def sigmoidDerivative(netVal):
    sigValue = sigmoid(netVal)
    return sigValue*(1-sigValue)

def tangetHyperbolicDerivative(netVal):
    tinhValue = tangetHyperbolic(netVal)
    return (-tinhValue)*tinhValue

layers = [2,2,2]
def forwardProp(X,Y,weights,layers,activation,*netVal):
    neurons = []
    neurons.append(X)
    for i in range(len(layers)-1):
        yHat = np.dot(neurons[i],weights[i])
        netVal.insert(0,yHat)
        yHat = activation[0]()
        neurons.append(yHat)

    return weights,neurons

def backword(weights,neurons,Y,activation,netVal):
    errorSignal = []
    yHatOutputLayer = neurons.pop()

    if activation=='sigmoid': d=sigmoidDerivative(netVal[0])
    else : d = tangetHyperbolicDerivative(netVal[0])

    outputLayer = (Y-yHatOutputLayer)*d
    errorSignal.insert(outputLayer)
    weightsRev = weights[::-1]
    for i in range(len(weights)-1):
        if activation == 'sigmoid':
            d = sigmoidDerivative(netVal[i+1])
        else:
            d = tangetHyperbolicDerivative(netVal[i+1])
        gradient = (np.dot(weightsRev[i],errorSignal[0]))*d
        errorSignal.insert(0,gradient)

    return errorSignal.reverse()

def updateWeights(errorSignal,weights,learn_rate,neurons):
    weightsHat=[]
    for i in range(len(weights)):
        weightsHat[i] = weights[i]+(learn_rate*neurons[i]*errorSignal[i])
    return weightsHat


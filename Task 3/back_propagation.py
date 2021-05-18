import numpy as np


def tangetHyperbolic(netVal):
    output = (1-np.exp(-netVal))/(1+np.exp(-netVal))
    return output

def sigmoid(netVal):
    output = []
    for i in range(len(netVal)):
        output.append (1/(1 + np.exp(-netVal[i])))
    return output
def sigmoidDerivative(netVal):
    sigValue = sigmoid(netVal)
    return sigValue*(1-sigValue)

def tangetHyperbolicDerivative(netVal):
    tinhValue = tangetHyperbolic(netVal)
    return (-tinhValue)*tinhValue

layers = [2,2,2]
def forwardProp(X,Y,weights,layers,activation,netVal,bias):
    neurons = []
    neurons.append(X)
    for i in range(len(layers)-1):
        neuron = np.array(neurons[i].reshape(neurons[i].shape[0], 1))
        yHat = np.dot(neuron.T,weights[i])
        netVal.clear()
        netVal.append(yHat)
        yHat = activation[0]()
        # if activation == 0:
        #     yHat = sigmoid(yHat[0])
        # else:
        #     yHat = tangetHyperbolicDerivative(netVal[i + 1])
        if bias:
            yHat.append(1)
        newY = np.array(yHat)
        neurons.append(newY)

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


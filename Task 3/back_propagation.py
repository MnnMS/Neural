import numpy as np


def tangetHyperbolic(netVal):
    output = []
    for i in range(len(netVal)):
        output.append(np.tanh(netVal[i]))
    return output

def sigmoid(netVal):
    output = []
    for i in range(len(netVal)):
        output.append (1/(1 + np.exp(-netVal[i])))
    return output
def sigmoidDerivative(neurons):
    output = []
    for i in range(len(neurons)):
        output.append(neurons[i]*(1-neurons[i]))
    return output

def tangetHyperbolicDerivative(neurons):
    output = []
    for i in range(len(neurons)):
        output.append((-neurons[i])*neurons[i])
    return output

layers = [2,2,2]
def forwardProp(X,weights,layers,activation,netVal,bias):
    neurons = []
    neurons.append(X)
    for i in range(len(layers)-1):
        neuron = np.array(neurons[i].reshape(neurons[i].shape[0], 1))
        yHat = np.dot(neuron.T,weights[i])
        netVal.clear()
        netVal.append(yHat)
        yHat = activation[0]()
        if bias and i!= len(layers) - 2 :
            yHat.append(1)
        newY = np.array(yHat)
        neurons.append(newY)

    return weights,neurons

def backword(weights,neurons,Y,activ_fun,bias):
    errorSignal = []
    yHatOutputLayer = neurons.pop()
    if activ_fun==0: d=sigmoidDerivative(yHatOutputLayer)
    else : d = tangetHyperbolicDerivative(yHatOutputLayer)
    outputLayer = (Y-yHatOutputLayer)*d
    errorSignal.insert(0,outputLayer)
    weightsRev = weights[::-1]
    neuronsRev = neurons[::-1]
    for i in range(len(weights)-1):
        if activ_fun == 0:
            d = sigmoidDerivative(neuronsRev[i])
        else:
            d = tangetHyperbolicDerivative(neuronsRev[i])
        if bias ==1:
            weightsRev[i]=weightsRev[i][:-1,:]
            d = d[:-1]
        gradient = (np.dot(weightsRev[i],errorSignal[0]))*d
        errorSignal.insert(0,gradient)

    return errorSignal

def updateWeights(errorSignal,weights,learn_rate,neurons):
    weightsHat=[]
    for i in range(len(weights)):
        e = np.array(errorSignal[i].reshape(1,errorSignal[i].shape[0]))
        n= np.array(neurons[i].reshape(neurons[i].shape[0],1))
        weightsHat.append(weights[i]+(learn_rate*(n*e)))
    return weightsHat


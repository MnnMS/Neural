import numpy as np


def tangetHyperbolic(netVal):

    return netVal

def sigmoid(netVal):
    output = 1/(1 + np.exp(-netVal))
    return output


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

w = forwardProp(layers)
#print(np.dot(w[0]))
print(w)
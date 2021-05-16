import numpy as np

def sigmoid(netVal):
    output = 1/(1 + np.exp(-netVal))
    return output


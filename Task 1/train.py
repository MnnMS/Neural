import numpy as np

def signum(x):
    return 0

def perceptron(epochs, X, T, bias, rate):
    W = np.random.rand([1, 3])
    if not bias:
        W[0] = 0
    while epochs:
        for i in range(0,X.shape[0]):
            net = np.dot(W.T, X[i])
            yhat = signum(net)
            if yhat == T[i]:
                continue
            else:
                L = T[i] - yhat
                W += rate * L * X[i]

        epochs -= 1

    return W
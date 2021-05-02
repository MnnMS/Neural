import numpy as np
import matplotlib.pyplot as plt


def signum(netVal):
    if netVal == 0:
        return 0
    else:
        return 1 if netVal > 0 else -1

def adaline(epochs, X, T, rate,mse,bias):
    newMse = 0
    m, _ = X.shape
    W = np.random.rand(3, 1)
    if not bias:
        W[0][0] = 0
    while epochs:
        for i in range(m):
            yhat = np.dot(W.T, X[i])
            if yhat == T[i]:
                continue
            else:
                L = T[i] - yhat
                W = W + (np.multiply(rate * L, X[i]).reshape(3, 1))

        for e in range(m):
            value = np.dot(W.T, X[e])
            newError = np.square(T[e] - value)
            newMse += newError
        newMse *= 1 / (2*m)
        if newMse <= mse:
            return W

        newMse = 0
        epochs -= 1

    return W

def drawLine(c1X, c1Y, c2X, c2Y, W, data):
    plt.scatter(c1X, c1Y)
    plt.scatter(c2X, c2Y)
    plt.xlabel('X{}'.format(data[2]+1))
    plt.ylabel('X{}'.format(data[3]+1))
    plt.legend([data[0],data[1]])

    w1 = W[1][0]
    w2 = W[2][0]
    b = W[0][0]
    y = np.multiply((-w1 / w2), c1X) - (b / w2)
    plt.plot(c1X, y, color='black', lw=4)
    plt.show()


def test(X, T, W):
    classes = np.unique(T)
    num_of_classes = classes.size
    confusion_matrix = np.zeros([num_of_classes, num_of_classes])

    for i in range(0, X.shape[0]):
        net = np.dot(W.T, X[i])
        yhat = signum(net)
        index = np.where(classes == T[i])
        if yhat == T[i]:
            confusion_matrix[index, index] += 1
        else:
            index_y = np.where(classes == yhat)
            confusion_matrix[index, index_y] += 1

    accuracy = (sum(np.diagonal(confusion_matrix)))/40

    return confusion_matrix, accuracy
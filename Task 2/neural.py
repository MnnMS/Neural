import numpy as np
import matplotlib.pyplot as plt

def signum(netVal):
    if netVal == 0:
        return 0
    else:
        return 1 if netVal > 0 else -1

def adaline(epochs, X, T, rate,mse, bias):
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

def drawLine(X_test, W, data):
    w1 = W[1][0]
    w2 = W[2][0]
    b = W[0][0]
    x = [np.min(X_test[:, [1,2]]), np.max(X_test[:, [1,2]])]
    y = [(-1 * w1 * x[0] - b) / w2, (-1 * w1 * x[1] - b) / w2]
    drawClasses(X_test, data)
    plt.plot(x, y, color='black', lw=2.5)
    plt.show()

def drawClasses(X_test, data):
    C11, C12, C21, C22 = get_XY(X_test)
    plt.scatter(C11, C12)
    plt.scatter(C21, C22)
    plt.xlabel('X{}'.format(data[2] + 1))
    plt.ylabel('X{}'.format(data[3] + 1))
    plt.legend([data[0], data[1]])
    plt.grid(lw=0.7, ls='--', alpha=0.8)

def get_XY(X_test):
    C11 = X_test[:20, 1]
    C12 = X_test[:20, 2]
    C21 = X_test[20:40, 1]
    C22 = X_test[20:40, 2]
    return C11, C12, C21, C22

def test(X, T, W):
    classes = np.unique(T)
    num_of_classes = classes.size
    confusion_matrix = np.zeros([num_of_classes, num_of_classes])
    total = X.shape[0]

    for i in range(0, total):
        net = np.dot(W.T, X[i])
        yhat = signum(net)
        index = np.where(classes == T[i])
        if yhat == T[i]:
            confusion_matrix[index, index] += 1
        else:
            index_y = np.where(classes == yhat)
            confusion_matrix[index, index_y] += 1

    accuracy = ((sum(np.diagonal(confusion_matrix)))/total) * 100
    return confusion_matrix.astype(int), accuracy
import numpy as np
import matplotlib.pyplot as plt

def signum(netVal):
    if netVal == 0:
        return 0
    else:
        return 1 if netVal > 0 else -1

def perceptron(epochs, X, T, bias, rate):
    W = np.random.rand(3, 1)
    if not bias:
        W[0][0] = 0
    while epochs:
        for i in range(60):
            netValue = np.dot(W.T, X[i])
            yhat = signum(netValue)
            if yhat == T[i]:
                continue
            else:
                L = T[i] - yhat
                k = rate * L * X[i]
                c = k.reshape(3,1)
                for j in range(3):
                    W[j][0] += c[j][0]
        epochs -= 1

    return W

def drawLine(dataset, W):
    arr = np.array(dataset)
    XS = arr[:150,0:4]
    mn = np.amin(XS)
    mx = np.amax(XS)
    w1 = W[1][0]
    w2 = W[2][0]
    b = W[0][0]
    for i in np.linspace(mn, mx):
        slope = -(b / w2)/(b / w1)
        inter = -(b/ w2)
        y = slope*i + inter
        plt.plot(i, y, 'ko')
    plt.show()

    # xj = -b / w2
    # P1 = [mn, xj]
    # xi = -b / w1
    # P2 = [xi, mx]
    # print(P1)
    # print(P2)
    # #plt.figure('fig')
    # plt.plot(P1, P2)

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

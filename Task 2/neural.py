import numpy as np
import matplotlib.pyplot as plt


def adaline(epochs, X, T, rate,mse):
    newMse = 0
    W = np.random.rand(3, 1)
    while epochs:
        for i in range(60):
            netValue = np.dot(W.T, X[i])
            yhat = netValue
            if yhat == T[i]:
                continue
            else:
                L = T[i] - yhat
                k = rate * L * X[i]
                c = k.reshape(3,1)
                for j in range(3):
                    W[j][0] += c[j][0]
        for e in range(60):
            value = np.dot(W.T, X[e])
            newError = np.square(T[e] - value)
            newMse += newError
            newMse *= 1 / 120
        if newMse <= mse:
            break
        newMse = 0
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


def test(X, T, W):
    classes = np.unique(T)
    num_of_classes = classes.size
    confusion_matrix = np.zeros([num_of_classes, num_of_classes])

    for i in range(0, X.shape[0]):
        net = np.dot(W.T, X[i])
        yhat = net
        index = np.where(classes == T[i])
        if yhat == T[i]:
            confusion_matrix[index, index] += 1
        else:
            index_y = np.where(classes == yhat)
            confusion_matrix[index, index_y] += 1

    accuracy = (sum(np.diagonal(confusion_matrix)))/40

    return confusion_matrix, accuracy
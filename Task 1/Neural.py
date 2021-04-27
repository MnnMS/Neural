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

def drawLine(dataset,class1, class2, W, f1, f2):
    arr = np.array(dataset)
    Class1 = arr[:50, :4]
    Class2 = arr[50:100, :4]
    Class3 = arr[100:150, :4]
    if class1 == 'Iris-setosa' and  class2 == 'Iris-versicolor':
        f1_C1 = Class1[:, f1]
        f1_C2 = Class2[:, f1]
        f2_C1 = Class1[:, f2]
        f2_C2 = Class2[:, f2]
    elif class1 == 'Iris-setosa' and  class2 == 'Iris-virginica':
        f1_C1 = Class1[:, f1]
        f1_C2 = Class3[:, f1]
        f2_C1 = Class1[:, f2]
        f2_C2 = Class3[:, f2]
    elif class2 == 'Iris-virginica' and  class1 == 'Iris-versicolor':
        f1_C1 = Class2[:, f1]
        f1_C2 = Class3[:, f1]
        f2_C1 = Class2[:, f2]
        f2_C2 = Class3[:, f2]

    w1 = W[1][0]
    w2 = W[2][0]
    b = W[0][0]

    plt.scatter(f1_C1, f2_C1)
    plt.scatter(f1_C2, f2_C2)
    plt.xlabel('X{}'.format(f1))
    plt.ylabel('X{}'.format(f2))
    plt.legend([class1, class2])

    print(W)
    slope = -(b / w2) / (b / w1) if b > 0 else -w2/w1
    inter = -(b / w2)
    y = slope*f1_C1 + inter
    plt.plot(f1_C1, y, color='red')
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

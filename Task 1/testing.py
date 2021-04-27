import train
import numpy as np


def test(X, T, W):
    classes = np.unique(T);
    num_of_classes = classes.size
    confusion_matrix = np.zeros([num_of_classes, num_of_classes])

    for i in range(0, X.shape[0]):
        net = np.dot(W.T, X[i])
        yhat = train.signum(net)
        index = np.where(classes == T[i])
        if yhat == T[i]:
            confusion_matrix[index, index] += 1
        else:
            index_y = np.where(classes == yhat)
            confusion_matrix[index, index_y] += 1

    accuracy = (sum(np.diagonal(confusion_matrix)))/40

    return confusion_matrix, accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def set_labels():
    dataset = pd.read_csv('IrisData.txt')
    newCol = []
    for i in range(0,len(dataset)):
        if dataset['Class'][i] == 'Iris-setosa':
            newCol.append([1, 0, 0])
        elif dataset['Class'][i] == 'Iris-versicolor':
            newCol.append([0, 1, 0])
        else:
            newCol.append([0, 0, 1])
    dataset.drop('Class', axis=1, inplace=True)
    dataset['Class'] = newCol
    return dataset

def extractFeatures(bias):
    dataset = set_labels()
    Data = np.array(dataset)
    X = Data[:, 0:4]
    T = Data[:, 4]
    b = np.ones([dataset.shape[0], 1])

    X_Train, X_Test, T_Train, T_Test, b_Train, b_Test = train_test_split(X, T, b, test_size=0.4, shuffle=True,stratify=T)

    if bias:
        X_Train = np.concatenate((b_Train, X_Train), axis=1)
        X_Test = np.concatenate((b_Test, X_Test), axis=1)

    return X_Train, X_Test, T_Train, T_Test

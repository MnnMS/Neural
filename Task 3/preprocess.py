import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    Class1 = Data[:50]
    np.random.shuffle(Class1)
    Class2 = Data[50:100]
    np.random.shuffle(Class2)
    Class3 = Data[100:150]
    np.random.shuffle(Class3)

    X = np.concatenate((Class1[:, 0:4], Class2[:, 0:4], Class3[:,0:4]))
    T = np.concatenate((Class1[:, 4], Class2[:, 4], Class3[:, 4]))
    b = np.ones([dataset.shape[0], 1])

    X_Train, _, T_Train, _, b_Train, b_Test = train_test_split(X, T, b, test_size=0.4)
    _, X_TestC1, _, T_TestC1 = train_test_split(X[:50, :], T[:50], test_size=0.4)
    _, X_TestC2, _, T_TestC2 = train_test_split(X[50:100, :], T[50:100], test_size=0.4)
    _, X_TestC3, _, T_TestC3 = train_test_split(X[100:150, :], T[100:150], test_size=0.4)

    X_Test = np.concatenate((X_TestC1, X_TestC2, X_TestC3))
    T_Test = np.concatenate((T_TestC1, T_TestC2, T_TestC3))

    arr = [0, 0, 0]
    for test in T_Test:
        i = test.index(1)
        arr[i] += 1

    if bias:
        X_Train = np.concatenate((b_Train, X_Train), axis=1)
        X_Test = np.concatenate((b_Test, X_Test), axis=1)

    return X_Train, X_Test, T_Train, T_Test

def extractFeatures2(bias=0):
    train_dataset = pd.read_csv('mnist_train.csv')
    test_dataset = pd.read_csv('mnist_test.csv')
    small_data = train_dataset.iloc[:30000, :]

    top_features = corr_matrix(small_data, 'label')
    train_dataset = train_dataset[top_features]
    test_dataset = test_dataset[top_features]

    train_Data = np.array(train_dataset)
    test_Data = np.array(test_dataset)
    np.random.shuffle(train_Data)
    np.random.shuffle(test_Data)

    X_Train = featureScaling(train_Data[:, 1:])
    T_Train = train_Data[:, 0]
    X_Test = featureScaling(test_Data[:, 1:])
    T_Test = test_Data[:, 0]

    if bias:
        b_Train = np.ones([train_Data.shape[0], 1])
        b_Test = np.ones([test_Data.shape[0], 1])
        X_Train = np.concatenate((b_Train, X_Train), axis=1)
        X_Test = np.concatenate((b_Test, X_Test), axis=1)

    return X_Train, X_Test, T_Train, T_Test

def corr_matrix(data, y_col_name, thresh=0.25):
    corr = data.corr()
    top_features = corr.index[corr[y_col_name] >= thresh]
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    return top_features

def featureScaling(X,a=0,b=1):
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        mx = max(X[:,i])
        mn = min(X[:,i])
        if mx != mn:
            Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(mx-mn))*(b-a)+a
    return Normalized_X

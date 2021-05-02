import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def set_labels(class1, class2):
    dataset = pd.read_csv('IrisData.txt')
    temp = dataset.replace({class1: 1, class2: -1}, value=None)
    return temp

def extractFeatures(dataset,class1,class2,f1,f2):
    Data = np.array(dataset)
    Class1 = Data[:50]
    np.random.shuffle(Class1)
    Class2 = Data[50:100]
    np.random.shuffle(Class2)
    Class3 = Data[100:150]
    np.random.shuffle(Class3)

    b = np.ones([100, 1])
    classA = []
    classB = []
    if class1 == 'Iris-setosa' and  class2 == 'Iris-versicolor':
        classA = Class1[:, [f1, f2]]
        classB = Class2[:, [f1, f2]]
        T = np.concatenate((Class1[:, 4], Class2[:, 4]))
    elif class1 == 'Iris-setosa' and  class2 == 'Iris-virginica':
        classA = Class1[:, [f1, f2]]
        classB = Class3[:, [f1, f2]]
        T = np.concatenate((Class1[:, 4], Class3[:, 4]))
    elif class2 == 'Iris-virginica' and  class1 == 'Iris-versicolor':
        classA = Class2[:, [f1, f2]]
        classB = Class3[:, [f1, f2]]
        T = np.concatenate((Class2[:, 4], Class3[:, 4]))

    X_TrainC1, X_TestC1, T_TrainC1, T_TestC1 = train_test_split(classA, T[:50], test_size=0.4)
    X_TrainC2, X_TestC2, T_TrainC2, T_TestC2 = train_test_split(classB, T[50:100], test_size=0.4)

    X_Train = np.concatenate((X_TrainC1, X_TrainC2))
    X_Train = np.concatenate((b[0:60], X_Train), axis=1)

    X_Test = np.concatenate((X_TestC1, X_TestC2))
    X_Test = np.concatenate((b[60:100], X_Test), axis=1)

    T_Train = np.concatenate((T_TrainC1, T_TrainC2))
    T_Test = np.concatenate((T_TestC1, T_TestC2))

    X_Train, T_Train = shuffleTrain(X_Train, T_Train)

    return X_Train, X_Test, T_Train, T_Test

def get_XY(class1, class2, f1, f2):
    C11 = class1[:, f1]
    C12 = class1[:, f2]
    C21 = class2[:, f1]
    C22 = class2[:, f2]
    return C11, C12, C21, C22

def shuffleTrain(X, T):
    train = list(zip(X, T))
    np.random.shuffle(train)
    X_train, T_train = zip(*train)
    return np.array(X_train), np.array(T_train)


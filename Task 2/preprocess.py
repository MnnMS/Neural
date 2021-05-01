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
    classXY = []
    if class1 == 'Iris-setosa' and  class2 == 'Iris-versicolor':
        X = np.concatenate((Class1[:, [f1, f2]], Class2[:, [f1, f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class1[:, 4], Class2[:, 4]))
        C11, C12, C21, C22 = get_XY(Class1, Class2, f1, f2)
    elif class1 == 'Iris-setosa' and  class2 == 'Iris-virginica':
        X = np.concatenate((Class1[:, [f1, f2]], Class3[:, [f1, f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class1[:, 4], Class3[:, 4]))
        C11, C12, C21, C22 = get_XY(Class1, Class3, f1, f2)
    elif class2 == 'Iris-virginica' and  class1 == 'Iris-versicolor':
        X = np.concatenate((Class2[:,[f1,f2]], Class3[:, [f1,f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class2[:, 4], Class3[:, 4]))
        C11, C12, C21, C22 = get_XY(Class2, Class3, f1, f2)

    X_Train, X_Test, T_Train, T_Test = train_test_split(X, T, test_size=0.4)
    classXY = [C11, C12, C21, C22]

    return X_Train, X_Test, T_Train, T_Test, classXY

def get_XY(class1, class2, f1, f2):
    C11 = class1[:, f1]
    C12 = class1[:, f2]
    C21 = class2[:, f1]
    C22 = class2[:, f2]
    return C11, C12, C21, C22

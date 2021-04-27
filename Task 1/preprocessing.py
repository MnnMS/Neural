import matplotlib.pyplot as plt
import numpy as np

def replace(dataset, cl1, cl2):
    if cl1 == 'Iris-setosa' and cl2 == 'Iris-versicolor':
        for i in range(150):
            if dataset["Class"][i] == "Iris-setosa":
                dataset["Class"][i] = 1
            if dataset["Class"][i] == "Iris-versicolor":
                dataset["Class"][i] = -1

    elif cl1 == 'Iris-setosa' and cl2 == 'Iris-virginica':
        for i in range(150):
            if dataset["Class"][i] == "Iris-setosa":
                dataset["Class"][i] = 1
            if dataset["Class"][i] == "Iris-virginica":
                dataset["Class"][i] = -1

    elif cl1 == 'Iris-versicolor' and cl2 == 'Iris-virginica':
        for i in range(150):
            if dataset["Class"][i] == "Iris-versicolor":
                dataset["Class"][i] = 1
            if dataset["Class"][i] == "Iris-virginica":
                dataset["Class"][i] = -1

def draw(dataset, x, y):
    arr = np.array(dataset)
    X11 = arr[:50, :4]
    X21 = arr[50:100, :4]
    X31 = arr[100:150, :4]

    plt.figure('fig')
    plt.scatter(X11[:, x], X11[:, y])
    plt.scatter(X21[:, x], X21[:, y])
    plt.scatter(X31[:, x], X31[:, y])

    plt.xlabel('X%i'.format(x+1))
    plt.ylabel('X%i'.format(y+1))
    #plt.show()

def extractFeatures(dataset,class1,class2,f1,f2,trainFlag):
    Data = np.array(dataset)
    Class1 = Data[:50]
    np.random.shuffle(Class1)
    Class2 = Data[50:100]
    np.random.shuffle(Class2)
    Class3 = Data[100:150]
    np.random.shuffle(Class3)

    start = 0 if trainFlag else 30
    end = 30 if trainFlag else 50
    size = 60 if trainFlag else 40

    b = np.ones([size, 1])
    if class1 == 'Iris-setosa' and  class2 == 'Iris-versicolor':
        X = np.concatenate((Class1[start:end, [f1, f2]], Class2[start:end, [f1, f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class1[start:end, 4], Class2[start:end, 4]))
    elif class1 == 'Iris-setosa' and  class2 == 'Iris-virginica':
        X = np.concatenate((Class1[start:end, [f1, f2]], Class3[start:end, [f1, f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class1[start:end, 4], Class3[start:end, 4]))
    elif class2 == 'Iris-virginica' and  class1 == 'Iris-versicolor':
        X = np.concatenate((Class2[start:end,[f1,f2]], Class3[start:end, [f1,f2]]))
        X = np.concatenate((b, X), axis=1)
        T = np.concatenate((Class2[start:end, 4], Class3[start:end, 4]))

    return X,T
import matplotlib as plt
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


    if x == 'X1' and y == 'X2':
        plt.figure('fig')
        plt.scatter(X11[:, 0], X11[:, 1])
        plt.scatter(X21[:, 0], X21[:, 1])
        plt.scatter(X31[:, 0], X31[:, 1])


    elif x == 'X1' and y == 'X3':
        plt.figure('fig')
        plt.scatter(X11[:, 0], X11[:, 2])
        plt.scatter(X21[:, 0], X21[:, 2])
        plt.scatter(X31[:, 0], X31[:, 2])

    elif x == 'X1' and y == 'X4':
        plt.figure('fig')
        plt.scatter(X11[:, 0], X11[:, 3])
        plt.scatter(X21[:, 0], X21[:, 3])
        plt.scatter(X31[:, 0], X31[:, 3])

    elif x == 'X2' and y == 'X3':
        plt.figure('fig')
        plt.scatter(X11[:, 1], X11[:, 2])
        plt.scatter(X21[:, 1], X21[:, 2])
        plt.scatter(X31[:, 1], X31[:, 2])

    elif x == 'X2' and y == 'X4':
        plt.figure('fig')
        plt.scatter(X11[:, 1], X11[:, 3])
        plt.scatter(X21[:, 1], X21[:, 3])
        plt.scatter(X31[:, 1], X31[:, 3])

    elif x == 'X3' and y == 'X4':
        plt.figure('fig')
        plt.scatter(X11[:, 2], X11[:, 3])
        plt.scatter(X21[:, 2], X21[:, 3])
        plt.scatter(X31[:, 2], X31[:, 3])

    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
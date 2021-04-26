import pandas as pd
import matplotlib.pyplot as plt
import train
import random
import numpy as np

dataset = pd.read_csv('IrisData.txt')
#print(dataset)



# Todo 1.replace class names with 0,1,2

def replace(cl1, cl2):
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




# Todo 2.draw Iris data
def draw(x, y):
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


draw('X1', 'X3')


# Todo 3.GUI

# Todo 4.Call perceptron
    # todo 4.1 extract features X(x0'bias',x1,x2) and their class T.
    # W = train.perceptron(0,0,0,1,0.1)
    # todo 4.2 draw classification Line.
    # w1 = W[1]
    # w2 = W[2]
    # b = W[0]
    # xj = -b / w2
    # P1 = [0, xj]
    # xi = -b / w1
    # P2 = [xi, 0]
Data =  np.array(dataset)
Class1 = Data[:50]
random.shuffle(Class1)
Class2 = Data[50:100]
random.shuffle(Class2)
Class3 = Data[100:150]
random.shuffle(Class3)
b = np.ones([50,1])
if(class1 ,class2):
        X = [b[:30] ,Class1[:30,x1], Class2[:30,x2]]
        T = [Class1[:30,4], Class2[:30,4]]
elif (class1, class3):
        X = [b[:30] ,Class1[:30,x1], Class3[:30,x2]]
        T = [Class1[:30,4], Class3[:30,4]]
elif (class2, class3):
        X = [b[:30] ,Class2[:30,x1], Class3[:30,x2]]
        T = [Class2[:30,4], Class3[:30,4]]
# w = train.perceptron(epochs,X,T,biasFlag,LearnRate)
# todo 4.2 draw classification Line.

# Todo 5.Test
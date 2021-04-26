import pandas as pd
import matplotlib.pyplot as plt
import train
import numpy as np
import random

dataset = pd.read_csv('IrisData.txt')
#print(dataset)

# Todo 1.replace class names with 0,1,2

# Todo 2.draw Iris data

# Todo 3.GUI

# Todo 4.Call perceptron
    # todo 4.1 extract features X(x0'bias',x1,x2) and their class T.
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
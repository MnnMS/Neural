import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import tkinter as tk
import Neural
import random
import numpy as np
import preprocessing


dataset = pd.read_csv('IrisData.txt')
#print(dataset)

# Todo 1.GUI

def Train():
    preprocessing.replace(dataset, class1, class2)
    X, T = preprocessing.extractFeatures(dataset, class1, class2, f1_combo.current(), f2_combo.current())
    W = Neural.perceptron(epochs_txt,X,T,var.get(),lrnRate_txt)
    preprocessing.draw(dataset, f1_combo.get(), f2_combo.get())



mainForm = Tk()
mainForm.geometry("650x500")
mainForm.title("Task 1")

f1_label = Label(mainForm,text = "Select Feature 1").place(x = 5, y = 10)
f2_label = Label(mainForm,text ="Select Feature 2").place(x = 5, y = 50)
clss_label = Label(mainForm,text ="Select Classes").place(x = 5, y = 100)
lrnRate_label = Label(mainForm,text ="Enter Learning Rate").place(x = 5, y = 150)
epochs_label = Label(mainForm,text ="Enter number of epochs").place(x = 5, y = 200)


features = ('X1','X2','X3','X4')
f1_combo = ttk.Combobox(mainForm,width = 10, values = features)
f1_combo.place(x = 100, y = 10)
f2_combo = ttk.Combobox(mainForm, width = 10, values = features)
f2_combo.place(x = 100, y = 50)
classes = ['C1 and C2', 'C1 and C3', 'C2 and C3']

clss_combo = ttk.Combobox(mainForm,width=10, values = classes)
clss_combo.place(x = 100, y = 100)
lrnRate_txt = Entry(mainForm).place(x = 150, y = 150)
epochs_txt = Entry(mainForm).place(x = 150, y = 200)
var = tk.IntVar()
bias_check = Checkbutton(mainForm,text = "Bias",variable=var)
bias_check.place(x = 5, y = 250)


train_button = Button(mainForm,text = "Train",command = Train).place(x = 300, y = 300)
#test_button = Button(mainForm,text = "Test",command = test).place(x = 400, y = 300)

if clss_combo.current() == 0:
    class1 = 'Iris-setosa'
    class2 = 'Iris-versicolor'
elif clss_combo.current() == 1:
    class1 = 'Iris-setosa'
    class2 = 'Iris-virginica'
else:
    class1 = 'Iris-versicolor'
    class2 = 'Iris-virginica'


mainForm.mainloop()



# Todo 1.replace class names with -1, 1

# Todo 2.draw Iris data




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

# w = train.perceptron(epochs,X,T,biasFlag,LearnRate)
# todo 4.2 draw classification Line.

# Todo 5.Test
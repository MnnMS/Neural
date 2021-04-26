import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import Neural
import random
import numpy as np
import preprocessing

dataset = pd.read_csv('IrisData.txt')
#print(dataset)

# Todo 1.GUI
def Train():
    print(f1_combo.get())
def test():
    print("test")

#tr = train()
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
bias_check = Checkbutton(mainForm,text = "Bias").place(x = 5, y = 250)

train_button = Button(mainForm,text = "Train",command = Train).place(x = 300, y = 300)
test_button = Button(mainForm,text = "Test",command = test).place(x = 400, y = 300)




mainForm.mainloop()



# Todo 1.replace class names with -1, 1
preprocessing.replace(dataset, 'setosa', 'versica')
# Todo 2.draw Iris data
preprocessing.draw(dataset, 'X1', 'X3')



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
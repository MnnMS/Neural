import pandas as pd
from tkinter import *
from tkinter import ttk
import tkinter as tk
import Neural
import preprocessing
import numpy as np
import tkinter.messagebox

dataset = pd.read_csv('IrisData.txt')
X_test = np.array([])
T_test = np.array([])
W_test = np.array([])

def Train():
    if clss_combo.current() == 0:
        class1 = 'Iris-setosa'
        class2 = 'Iris-versicolor'
    elif clss_combo.current() == 1:
        class1 = 'Iris-setosa'
        class2 = 'Iris-virginica'
    else:
        class1 = 'Iris-versicolor'
        class2 = 'Iris-virginica'

    preprocessing.replace(dataset, class1, class2)
    X, T = preprocessing.extractFeatures(dataset, class1, class2, f1_combo.current(), f2_combo.current(), trainFlag=True)
    W = Neural.perceptron(int(epochs_txt.get()),X,T,var.get(),float(lrnRate_txt.get()))
    #print(W)
    preprocessing.draw(dataset, f1_combo.get(), f2_combo.get())
    Neural.drawLine(dataset, W)

    # testing
    X, T = preprocessing.extractFeatures(dataset, class1, class2, f1_combo.current(), f2_combo.current(),
                                        trainFlag=False)
    global X_test
    X_test=X
    global T_test
    T_test= T
    global W_test
    W_test = W


def test():
    if X_test.size == 0 and T_test.size == 0 and W_test.size == 0:
        tk.messagebox.showinfo(title=None, message="Please train data before testing")
    else:
        matrix, accuracy = Neural.test(X_test, T_test, W_test)
        print("confusion Matrix = \n",matrix)
        print("Accuracy = ",accuracy)


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
lrnRate_var = tk.IntVar()
lrnRate_txt = Entry(mainForm)
lrnRate_txt.place(x = 150, y = 150)
epoch_var = tk.IntVar()
epochs_txt = Entry(mainForm)
epochs_txt.place(x = 150, y = 200)
var = tk.IntVar()
bias_check = Checkbutton(mainForm,text = "Bias",variable=var)
bias_check.place(x = 5, y = 250)


train_button = Button(mainForm,text = "Train",command = Train).place(x = 300, y = 300)
test_button = Button(mainForm,text = "Test",command = test).place(x = 400, y = 300)

mainForm.mainloop()

from tkinter import *
from tkinter import ttk
import tkinter as tk
import neural
import preprocess
import numpy as np
import tkinter.messagebox

X_test = np.array([])
T_test = np.array([])
W_test = np.array([])
Classes = {0:['Iris-setosa', 'Iris-versicolor'], 1:['Iris-setosa', 'Iris-virginica'], 2:['Iris-versicolor', 'Iris-virginica']}

def Train():
    #preprocess
    class1, class2 = Classes[clss_combo.current()]
    dataset = preprocess.set_labels(class1, class2)
    X_Train, X_Test, T_Train, T_Test, classXY = preprocess.extractFeatures(dataset,class1,class2,f1_combo.current(),f2_combo.current())

    # train
    W = neural.adaline(int(epochs_txt.get()),X_Train,T_Train,float(lrnRate_txt.get()),float(mse_txt.get()),var.get())
    diagramData = [class1, class2, f1_combo.current(), f2_combo.current()]
    #neural.drawLine(classXY[0], classXY[1], classXY[2], classXY[3], W, diagramData)
    neural.drawLine2(X_Test, W, diagramData)

    # test
    global X_test
    X_test= X_Test
    global T_test
    T_test= T_Test
    global W_test
    W_test = W

def test():
    if X_test.size == 0 and T_test.size == 0 and W_test.size == 0:
        tk.messagebox.showinfo(title=None, message="Please train data before testing")
    else:
        matrix, accuracy = neural.test(X_test, T_test, W_test)
        print("confusion Matrix = \n",matrix)
        print("Accuracy = ",accuracy)

mainForm = Tk()
mainForm.geometry("650x500")
mainForm.title("Task 2")

f1_label = Label(mainForm,text = "Select Feature 1").place(x = 5, y = 10)
f2_label = Label(mainForm,text ="Select Feature 2").place(x = 5, y = 50)
clss_label = Label(mainForm,text ="Select Classes").place(x = 5, y = 100)
lrnRate_label = Label(mainForm,text ="Enter Learning Rate").place(x = 5, y = 150)
epochs_label = Label(mainForm,text ="Enter number of epochs").place(x = 5, y = 200)
mse_label = Label(mainForm,text ="Enter MSE").place(x = 5, y = 250)

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
mse_var = tk.IntVar()
mse_txt = Entry(mainForm)
mse_txt.place(x = 150, y = 250)
var = tk.IntVar()
bias_check = Checkbutton(mainForm,text = "Bias",variable=var)
bias_check.place(x = 5, y = 300)

train_button = Button(mainForm,text = "Train",command = Train).place(x = 300, y = 300)
test_button = Button(mainForm,text = "Test",command = test).place(x = 400, y = 300)

mainForm.mainloop()

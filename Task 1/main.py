import pandas as pd
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import train

dataset = pd.read_csv('IrisData.txt')
print(dataset)
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

# Todo 1.replace class names with 0,1,2

# Todo 2.draw Iris data

# Todo 3.GUI

# Todo 4.Call perceptron
    # todo 4.1 extract features X(x0'bias',x1,x2) and their class T.
    # w = train.perceptron(epochs,X,T,biasFlag,LearnRate)
    # todo 4.2 draw classification Line.

# Todo 5.Test
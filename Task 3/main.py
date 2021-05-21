from tkinter import *
from tkinter import ttk
import tkinter as tk
import random
from preprocess import extractFeatures2,extractFeatures
import numpy as np
import train
import test

def set_text(text, txtBox):
    txtBox.delete(0,END)
    txtBox.insert(0,text)
    return

def autoFill():
    layers = random.randrange(2,5)
    neurons = np.random.randint(2, 15, layers)
    funct = random.randrange(2)
    ep = random.randrange(300,1000)
    set_text(str(layers), layers_txt)
    set_text("0.01", lrnRate_txt)
    set_text(str(ep), epochs_txt)
    neur_txt = ""
    for i in range(layers):
        neur_txt += str(neurons[i])
        if i != layers-1:
            neur_txt += ','
    set_text(neur_txt,neu_txt)
    funcs_combo.set(functions[funct])
    bias_check.select()
    return 

def Train():
    fun_sel = funcs_combo.current()
    num_layers = int(layers_txt.get())
    neurons = list(str(neu_txt.get()).split(','))
    for i in range(0,len(neurons)):
        neurons[i] = int(neurons[i])
    epochs_val = int(epochs_txt.get())
    lrnRate_val = float(lrnRate_txt.get())
    bias_val = int(bias_var.get())
    bonus_val = int(bonus_var.get())

    #preprocess
    X_Train, X_Test, T_Train, T_Test = extractFeatures2(bias_val) if bonus_val else extractFeatures(bias_val)
    
    # train
    W,nu,layers,activ,netVal = train.train(X_Train, T_Train, neurons, epochs_val, lrnRate_val, bias_val, fun_sel)

    # Test
    matrix,accuracy = test.test(X_Test,T_Test,W,layers,activ,netVal,bias_val,bonus_val)
    print("confusion Matrix = \n", matrix)
    print(accuracy)


mainForm = Tk()
mainForm.geometry("350x300")
mainForm.title("Task 3")

f1_label = Label(mainForm,text = "Select Function").place(x = 35, y = 10)
f2_label = Label(mainForm,text ="Enter # of Layers").place(x = 35, y = 50)
lrnRate_label = Label(mainForm,text ="Enter ETA").place(x = 35, y = 170)
epochs_label = Label(mainForm,text ="Enter number of epochs").place(x = 35, y = 130)
neu_label = Label(mainForm,text ="Enter # of Neurons").place(x = 35, y = 90)

functions = ('Sigmoid','Hyper Tanget')
funcs_combo = ttk.Combobox(mainForm,width = 17, values = functions)
funcs_combo.place(x = 180, y = 10)

layers_txt = Entry(mainForm)
layers_txt.place(x = 180, y = 50)

lrnRate_txt = Entry(mainForm)
lrnRate_txt.place(x = 180, y = 170)

epochs_txt = Entry(mainForm)
epochs_txt.place(x = 180, y = 130)

neu_txt = Entry(mainForm)
neu_txt.place(x = 180, y = 90)

bias_var = tk.IntVar()
bias_check = Checkbutton(mainForm,text = "Bias",variable=bias_var)
bias_check.place(x = 35, y = 210)

bonus_var = tk.IntVar()
bonus_check = Checkbutton(mainForm,text = "Bonus",variable=bonus_var)
bonus_check.place(x = 100, y = 210)

autoFill_button = Button(mainForm,text = "AutoFill",command = autoFill, width=17, bd=1, bg='#bdc3c7').place(x = 180, y = 210)
train_button = Button(mainForm,text = "Run",command = Train, width=20).place(x = 90, y = 255)

mainForm.resizable(False, False)

mainForm.mainloop()

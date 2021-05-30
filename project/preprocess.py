from tqdm import tqdm
import os
import cv2
from model_paras import *
import numpy as np
from random import shuffle

def create_train_data():
    data = []
    for img in tqdm(os.listdir(TRAIN_DIR_A)):
        path = os.path.join(TRAIN_DIR_A, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img_data), np.array([1, 0])])

    for img in tqdm(os.listdir(TRAIN_DIR_N)):
        path = os.path.join(TRAIN_DIR_N, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img_data), np.array([0, 1])])

    shuffle(data)
    np.save(TRAIN_DIR, data)
    return data

def create_test_data(model):
    data = []
    names = []
    for img in tqdm(os.listdir(TEST_DATA_DIR)):
        names.append(img)
        path = os.path.join(TEST_DATA_DIR, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        pred = model.predict([img_data])[0]
        data.append(pred)
    return data, names

def get_dataset():
    if os.path.exists(TRAIN_DIR):
        data = np.load(TRAIN_DIR, allow_pickle=True)
    else:
        data = create_train_data()

    train_size = int(TRAIN_VALID_SPLIT * len(data))
    train = data[:train_size]
    test = data[train_size:]

    X_Train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #print(X_Train.shape)
    X_Test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #print(X_Test.shape)
    Y_Train = [i[1] for i in train]
    Y_Test = [i[1] for i in test]


    return X_Train, X_Test, Y_Train, Y_Test

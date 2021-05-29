from tqdm import tqdm
import os
import cv2
from model_paras import *
import numpy as np
from sklearn.model_selection import train_test_split

def create_train_data():
    X = []
    Y = []
    for img in tqdm(os.listdir(TRAIN_DIR_A)):
        path = os.path.join(TRAIN_DIR_A, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img_data))
        Y.append([1, 0])
    for img in tqdm(os.listdir(TRAIN_DIR_N)):
        path = os.path.join(TRAIN_DIR_N, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img_data))
        Y.append([0, 1])
    np.save(TRAIN_DIR, X)
    np.save(TEST_DIR, Y)
    return X, Y

def get_dataset():
    if os.path.exists(TRAIN_DIR):
        X = np.load(TRAIN_DIR)
        Y = np.load(TEST_DIR)
    else:
        X, Y = create_train_data()

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=TRAIN_VALID_SPLIT, stratify=Y, shuffle=True)
    return X_Train, X_Test, Y_Train, Y_Test

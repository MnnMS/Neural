import tensorflow as tf
import numpy as np
import tflearn
import os
import pandas as pd
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from preprocess import get_dataset, create_test_data
from model_paras import *

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


X_Train, X_Test, Y_Train, Y_Test = get_dataset()
X_Train = np.array(X_Train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_Test = np.array(X_Test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 2, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta') and not FORCE_TRAIN:
    model.load('./model.tfl')
else:
    model.fit({'input': X_Train}, {'targets': Y_Train}, n_epoch=EPOCHS,
          validation_set=({'input': X_Test}, {'targets': Y_Test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

test_acc = model.evaluate(X_Test,  Y_Test)
print(test_acc)

file = pd.read_csv("Submit.csv")
preds, names = create_test_data(model)
file['Image'] = names
output = []
for pred in preds:
    if pred[0] > pred[1]:
        output.append(1)
    else:
        output.append(0)
file['Label'] = output
file.to_csv("Submit.csv")


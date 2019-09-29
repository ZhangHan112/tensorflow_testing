"""
利用 AlexNet 訓練 MNIST dataset
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

batch_size = 64
num_classes = 10
epochs = 10
img_shape = (28,28,1)

# input dimensions
img_rows, img_cols = 28,28

# dataset input
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# data initialization
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Define the input layer
inputs = keras.Input(shape = [img_rows, img_cols, 1])

 #Define the converlutional layer 1
conv1 = keras.layers.Conv2D(filters= 64, kernel_size= [11, 11], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(inputs)
# Define the pooling layer 1
pooling1 = keras.layers.AveragePooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'valid')(conv1)
# Define the standardization layer 1
stand1 = keras.layers.BatchNormalization(axis= 1)(pooling1)

# Define the converlutional layer 2
conv2 = keras.layers.Conv2D(filters= 192, kernel_size= [5, 5], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(stand1)
# Defien the pooling layer 2
pooling2 = keras.layers.AveragePooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'valid')(conv2)
# Define the standardization layer 2
stand2 = keras.layers.BatchNormalization(axis= 1)(pooling2)

# Define the converlutional layer 3
conv3 = keras.layers.Conv2D(filters= 384, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(stand2)
stand3 = keras.layers.BatchNormalization(axis=1)(conv3)

# Define the converlutional layer 4
conv4 = keras.layers.Conv2D(filters= 384, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(stand3)
stand4 = keras.layers.BatchNormalization(axis=1)(conv4)

# Define the converlutional layer 5
conv5 = keras.layers.Conv2D(filters= 256, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(stand4)
pooling5 = keras.layers.AveragePooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'valid')(conv5)
stand5 = keras.layers.BatchNormalization(axis=1)(pooling5)

# Define the fully connected layer
flatten = keras.layers.Flatten()(stand5)
fc1 = keras.layers.Dense(4096, activation= keras.activations.relu, use_bias= True)(flatten)
drop1 = keras.layers.Dropout(0.5)(fc1)

fc2 = keras.layers.Dense(4096, activation= keras.activations.relu, use_bias= True)(drop1)
drop2 = keras.layers.Dropout(0.5)(fc2)

fc3 = keras.layers.Dense(10, activation= keras.activations.softmax, use_bias= True)(drop2)

# 基於Model方法構建模型
model = keras.Model(inputs= inputs, outputs = fc3)
# 編譯模型
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 訓練配置，僅供參考
model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs, validation_data=(x_test,y_test))
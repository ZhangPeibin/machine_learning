#!/usr/bin/env python
# coding=utf-8


import numpy as np
# Sequential 是一个神经网络层的线性栈
from keras.models import Sequential
# 导入一些核心层
from keras.layers import Dense, Dropout, Activation, Flatten
# 导入cnn层
from keras.layers import Convolution2D, MaxPool2D
# 导入工具
from keras.utils import np_utils
# mnist 是深度学习和计算机视觉入门的很好的数据集
from keras.datasets import mnist
# 绘图库
from matplotlib import pyplot as plt


np.random.seed(123)  # for reproducibility

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print x_train.shape

# plt.imshow(x_train[0])
# plt.show()
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print x_train.shape

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 定义模型结构
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)

print "Test loss: %s " % score[0]
print "Test accuracy: %s " % score[1]

# model.save("mnist.model")

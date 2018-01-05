#!/usr/bin/env python
# coding=utf-8

import random
import load_face_data as lfd
from sklearn.model_selection import train_test_split
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, \
    Activation, MaxPool2D, Dropout, Flatten, Dense
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

data_path = "./data/me/"
model_path = "./data/me.face.model.h5"


class DataSet:

    def __init__(self, path_name):

        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集的方法
    # @param image_rows     图片的行
    # @param image_cols     图片的宽
    # @param img_channels   图片的通道,rgb = 3
    # @param nb_classes     分类种数
    def load(self, image_rows=lfd.SIZE, image_cols=lfd.SIZE, img_channels=3,
             nb_classes=2):
        images, labels = lfd.load_data_set(self.path_name)

        # 通过sklearn获取我们的训练集，验证集数据
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels,
                                                                                  test_size=0.3,
                                                                                  random_state=random.randint(0, 100))

        _, test_images, _, test_labels = train_test_split(images, labels,
                                                          test_size=0.2,
                                                          random_state=random.randint(1, 100))
        # 根据keras的后端系统更改图像的维度顺序
        if backend.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, image_rows, image_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, image_rows, image_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, image_rows, image_cols)
            self.input_shape = (img_channels, image_cols, image_rows)
        else:
            train_images = train_images.reshape(train_images.shape[0], image_rows, image_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], image_rows, image_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], image_rows, image_cols, img_channels)
            self.input_shape = (image_cols, image_rows, img_channels)

        # 类别进行one-hot编码使其向量化
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        # 像素数据浮点化以便归一化
        train_images = train_images.astype("float32")
        test_images = test_images.astype("float32")
        valid_images = valid_images.astype("float32")

        # 将其归一化，图像的个像素值归一化到0到1之间
        train_images = train_images / 255
        valid_images = valid_images / 255
        test_images = test_images / 255

        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.test_images = test_images
        self.test_labels = test_labels

        print self.train_images.shape
        print self.valid_images.shape
        print self.test_images.shape


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, data, nb_classes=2):

        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加
        # 专业名称为序贯模型或者线性堆叠模型
        self.model = Sequential()

        # 添加一个二维卷积层,包含32个卷积核，每个卷积核大小为3*3
        self.model.add(Convolution2D(32, (3, 3), activation="relu",input_shape=data.input_shape))
        self.model.add(Convolution2D(32, (3, 3), activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))     # 添加池化层
        self.model.add(Dropout(0.25))                   # Dropout层

        # self.model.add(Convolution2D(64, (3, 3)))
        # self.model.add(Activation('relu'))

        # 将多维数据处理为単维数据
        self.model.add(Flatten())
        # 添加全连接层
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # 全连接层最终目的就是完成我们的分类要求：0或者1
        self.model.add(Dense(nb_classes))
        self.model.add(Activation("softmax"))

        # print self.model.summary()

    def train(self, data, batch_size=128, nb_epoch=3, data_augmentation=True):
        # 采用SGD+momentum的优化器进行训练
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        if not data_augmentation:
            self.model.fit(data.train_images,
                           data.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(data.valid_images, data.valid_labels),
                           shuffle=True)
        else:
            imagedatagenerator = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=60,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=False,
                vertical_flip=False,
                fill_mode='nearest'
            )

            imagedatagenerator.fit(data.train_images)
            self.model.fit_generator(imagedatagenerator.flow(
                data.train_images, data.train_labels,
                batch_size=batch_size),
                samples_per_epoch=data.train_images.shape[0],
                nb_epoch=nb_epoch,
                validation_data=(data.valid_images, data.valid_labels)
            )

    def evaluate(self, data):
        score = self.model.evaluate(data.test_images, data.test_labels, verbose=0)
        print score

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def face_predict(self, image):
        if image is None:
            return

        if backend.image_dim_ordering() == 'th' and image.shape != (1, 3, lfd.SIZE, lfd.SIZE):
            image = lfd.resize_img(image)
            image = image.reshape(1, 3, lfd.SIZE, lfd.SIZE)
        elif backend.image_dim_ordering() == 'tf' and image.shape != (1, lfd.SIZE, lfd.SIZE, 3):
            image = lfd.resize_img(image)
            image = image.reshape(1, lfd.SIZE, lfd.SIZE, 3)

        image = image.astype("float32")
        image /= 255

        result = self.model.predict_classes(image)

        return result[0]


if __name__ == '__main__':
    ds = DataSet(data_path)
    ds.load()

    model = Model()
    model.build_model(ds)
    model.train(ds, data_augmentation=False)
    model.save_model(model_path)
    model.evaluate(ds)

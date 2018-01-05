#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model


def read_img_by_pil(paths):
    d = []
    for path in paths:
        img = Image.open(path).convert("L")

        if img.size[0] != 28 or img.size[1] != 28:
            img = img.resize((28, 28))
        arr = []

        for i in range(28):
            for j in range(28):
                # mnist 里的颜色是0代表白色（背景），1.0代表黑色
                pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
                # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
                arr.append(pixel)

        arr1 = np.array(arr).reshape((1, 28, 28, 1))
        d.append(arr1)
    return d


def read_img_by_cv(paths):
    d = []
    for path in paths:
        # 直接读为灰度图像
        img = cv2.imread(path, 0)
        width = img.shape[0]
        height = img.shape[1]
        # 修改大小为21,21
        if width != 28 or height != 28:
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)

        # 阈值处理，将图片转为黑白两个颜色
        ret, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
        # 转为nparray
        imgdata = np.asarray(img, dtype='float32')
        # 格式花为tensorflow要的格式
        testdata = imgdata.reshape(1, 28, 28, 1)
        d.append(testdata)
    return d


model = load_model("mnist.model")
path = ["test.png"]
data = read_img_by_cv(path)
testdata = np.empty((len(data), 28, 28, 1), dtype='float32')
for d in range(len(data)):
    testdata[d, :, :, :] = data[d]

source = model.predict_classes(testdata, batch_size=1, verbose=0)
print source


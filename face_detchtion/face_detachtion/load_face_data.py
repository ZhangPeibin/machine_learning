#!/usr/bin/env python
# coding=utf-8

import cv2
import os
import numpy as np

SIZE = 64
imageArray = []
labelArray = []


def resize_img(img, width=SIZE, height=SIZE):
    return cv2.resize(img, (width, height))


def read_img(read_path):
    for imgpath in os.listdir(read_path):
        fullPath = os.path.abspath(os.path.join(read_path, imgpath))
        if os.path.isdir(imgpath):
            read_img(fullPath)
        else:
            if imgpath.endswith(".jpg"):
                img = cv2.imread(fullPath)
                if img is None:
                    continue

                imageArray.append(resize_img(img))
                labelArray.append(imgpath)

    return imageArray, labelArray


def load_data_set(data_path):
    images, labels = read_img(data_path)

    # 转成4维数组 (num,size,size,deep)
    # (2859, 64, 64, 3)
    imgdata = np.array(images)
    labeldata = np.array([0 if label.startswith('me') else 1 for label in labels])

    return imgdata, labeldata


if __name__ == '__main__':
    load_data_set("./me/")

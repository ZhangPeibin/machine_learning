#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("data/test.jpg", 'rb').read()
with tf.Session() as sess:
    # 将图片解码为一个三维矩阵，解码之后的结果为一个张量
    image_data = tf.image.decode_jpeg(image)
    # print image_data.eval()
    plt.imshow(image_data.eval())
    plt.show()
    # 调整图像大小
    resized = tf.image.resize_images(image_data, [64, 64], method=0)
    retype = tf.cast(resized, tf.uint8)
    plt.imshow(retype.eval())
    plt.show()

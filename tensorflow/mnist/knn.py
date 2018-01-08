#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from input_data import read_data_sets
import numpy as np

mnist = read_data_sets("../data/MNIST_data/", one_hot=True)

# 获取5000个训练集
train_x, train_y = mnist.train.next_batch(5000)
# 获取200个测试集
test_x, test_y = mnist.test.next_batch(200)

train_input = tf.placeholder(dtype=tf.float32, shape=[None, 784])
test_input = tf.placeholder(dtype=tf.float32, shape=[784])

test_input_negative = tf.negative(test_input)
distance = tf.reduce_sum(tf.abs(tf.add(train_input, test_input_negative)), reduction_indices=1)

pred = tf.argmin(distance, 0)
accuracy = 0

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(len(test_x)):
    feed_dict = {train_input: train_x, test_input: test_x[i, :]}
    nearies_index = sess.run(pred,feed_dict=feed_dict)
    #print u"测试数据:%s,预测分类:%s,真实类别:%s" % \
    #      (i, np.argmax(train_y[nearies_index]), np.argmax(test_y[i]))
    if i == 0:
        print sess.run(test_input_negative,feed_dict={test_input: test_x[i, :]})
        print test_x[i, :]
    if np.argmax(train_y[nearies_index]) == np.argmax(test_y[i]):
        accuracy += 1. / len(test_x)

print u"分类准确率为:%s" % accuracy

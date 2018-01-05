#!usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

# 利用Numpy生产假数据，总共100个点,二维数组
x_data = np.float32(np.random.rand(2, 100))

# 对二维数组求点积实际上是 一维 × 二维结果也为一维
y_data = np.dot([0.100, 0.200], x_data) + 0.03

# 构造一个线程模型
# 创建一个一维的数组，元素全部为0
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 矩阵相乘
y = tf.matmul(w, x_data) + b

# 最小方差
loss = tf.reduce_mean(tf.squeeze(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss=loss)

# 初始化全部变量
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(w), sess.run(b)

print sess.run(loss)

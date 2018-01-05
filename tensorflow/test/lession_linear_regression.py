#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
points_set = []
for i in range(num_points):
    x = np.random.normal(0.0, 0.54)
    y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)

    points_set.append([x, y])

x_data = [v[0] for v in points_set]
y_data = [v[1] for v in points_set]

# plt.plot(x_data, y_data, 'ro', label="Original data")
# plt.legend()
# plt.show()


# 使用tensorflow实现 y = nx+b 线性模型

# 我们所训练的真实数据的x
x = tf.placeholder(dtype=tf.float32)
# x的权重
n = tf.Variable(tf.zeros([1]))
# x的偏移值
b = tf.Variable(tf.zeros([1]))
# x所对应的真实数据y
y_ = tf.placeholder(dtype=tf.float32)

# 定义我们的线性方程
y = x * n + b

# 定义我们的损失函数
loss = tf.reduce_mean(tf.square(y-y_))

# 定义我们的梯度下降优化器,其核心目标是找到特定的变量使loss的损失为最小
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.InteractiveSession()
# 初始化我们的所有变量
sess.run(tf.global_variables_initializer())

for i in range(num_points):
    feed_dict = {x: x_data, y_: y_data}
    sess.run(train_step, feed_dict=feed_dict)
    print "w>> %f , b>> %f ,loss>> %f" % (sess.run(n), sess.run(b), sess.run(loss,feed_dict=feed_dict))

plt.plot(x_data, y_data, 'ro', label="Original data")
plt.plot(x_data, sess.run(n) * x_data + sess.run(b))
plt.legend()
plt.show()
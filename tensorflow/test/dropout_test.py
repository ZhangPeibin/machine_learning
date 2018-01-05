#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

sess = tf.InteractiveSession()

dropout = tf.placeholder(dtype=tf.float32)
x = tf.Variable(tf.ones([10, 10]))
y = tf.nn.dropout(x, dropout)
sess.run(tf.global_variables_initializer())

print sess.run(y, feed_dict={dropout: 0.4})


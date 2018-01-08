# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple MNIST classifier example with JIT XLA and timelines.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)
  print(mnist.train.images.shape)
  print(mnist.train.labels.shape)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  # Define loss and optimizer
  # y_ = tf.placeholder(tf.int64, [None])
  y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # logit outputs of 'y', and then average across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
  # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),reduction_indices=[1]))
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(0.05, global_step, 10, 0.98, staircase=True)

  train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy,global_step=global_step)

  config = tf.ConfigProto()
  jit_level = 0
  if FLAGS.xla:
    # Turns on XLA JIT compilation.
    jit_level = tf.OptimizerOptions.ON_1

  config.graph_options.optimizer_options.global_jit_level = jit_level
  run_metadata = tf.RunMetadata()
  sess = tf.Session(config=config)
  tf.global_variables_initializer().run(session=sess)
  # Train
  train_loops = 1000
  for i in range(train_loops):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # Create a timeline for the last loop and export to json to view with
    # chrome://tracing/.
    if i == train_loops - 1:
      sess.run(train_step,
               feed_dict={x: batch_xs,
                          y_: batch_ys},
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open('timeline.ctf.json', 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format())
    else:
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy,
                 feed_dict={x: mnist.test.images,
                            y_: mnist.test.labels}))
  sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--xla', type=bool, default=True, help='Turn xla via JIT on')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

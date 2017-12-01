import numpy as np
import os
import tensorflow as tf
import sys
import code
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt


from load import *


def prep(k):
    (x,y_,tx,ty_) = load_sample_dataset()
    n = x.shape[0] // k
    return (np.array_split(normalize(x),n), np.array_split(onehot(y_, 10),n), 0, n,
            normalize(tx), onehot(ty_, 10))

def normalize(x):
    x = x - np.mean(x, axis=(1,2))[..., np.newaxis, np.newaxis]
    x = x / np.linalg.norm(x, axis=(1,2))[..., np.newaxis, np.newaxis]
    return x

def onehot(y,l):
    len = y.shape[0]
    tmp = np.zeros((len,l));
    tmp[np.arange(len), np.squeeze(y)] = 1
    return tmp

def next_batch(d):
    (xs, y_s, i, n, tx, ty_) = d
    return ((xs[i], y_s[i]), (xs, y_s, (i+1)%n, n, tx, ty_))

def test_data(d):
    (_, _, _, _, tx, ty_) = d
    return (tx, ty_)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# (a), (b) & (c)

data = prep(50)


x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y_ = tf.placeholder(tf.int64, shape=[None, 10])
# y_onehot = tf.one_hot(y_, 10)

x_image = tf.reshape(x, [-1, 28, 28, 1])


# 1st convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# training
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_accuracy = np.zeros([201, 1], np.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20001):
        # batch = mnist.train.next_batch(50)
        ((X,Y_), data) = next_batch(data)
        if i % 100 == 0:
            train_accuracy[int(i/100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy[int(i/100)]))
        train_step.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

    (tX,tY_) = test_data(data)
    print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))
    W1 = sess.run(W_conv1)
    b1 = sess.run(b_conv1)
    W2 = sess.run(W_conv2)
    b2 = sess.run(b_conv2)
    Wfc = sess.run(W_fc1)
    bfc = sess.run(b_fc1)

# (d) remove the second convolutional layer
'''
W_fc_only1 = weight_variable([14 * 14 * 32, 1024])
h_pool2_flat_only1 = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1_only1 = tf.nn.relu(tf.matmul(h_pool2_flat_only1, W_fc_only1) + b_fc1)
h_fc1_drop_only1 = tf.nn.dropout(h_fc1_only1, keep_prob)
y_conv_only1 = tf.matmul(h_fc1_drop_only1, W_fc2) + b_fc2
cross_entropy_only1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_only1))
train_step_only1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_only1)
correct_prediction_only1 = tf.equal(tf.argmax(y_conv_only1, 1), tf.argmax(y_, 1))
accuracy_only1 = tf.reduce_mean(tf.cast(correct_prediction_only1, tf.float32))
train_accuracy_only1 = np.zeros([201, 1], np.float32)
with tf.Session() as sess_only1:
  sess_only1.run(tf.global_variables_initializer())
  for i in range(20001):
    # batch = mnist.train.next_batch(50)
    ((X,Y_), data) = next_batch(data)
    if i % 100 == 0:
      train_accuracy_only1[int(i/100)] = accuracy_only1.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy_only1[int(i/100)]))
    train_step_only1.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

  (tX,tY_) = test_data(data)
  print('test accuracy %g' % accuracy_only1.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))



plt.figure()
plt.plot(train_accuracy, "k", label="2 convolutional layers")
plt.plot(train_accuracy_only1, "r", label="removed the 1st convolutional layer")
plt.legend()
plt.show()
'''

# (e) Try different step sizes for Gradient Descent Optimizer
'''
train_step1 = tf.train.GradientDescentOptimizer(1e-6).minimize(cross_entropy)
train_step2 = tf.train.GradientDescentOptimizer(5e-3).minimize(cross_entropy)
train_step3 = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
train_step4 = tf.train.GradientDescentOptimizer(9e-1).minimize(cross_entropy)
train_accuracy1 = np.zeros([201, 1], np.float32)
train_accuracy2 = np.zeros([201, 1], np.float32)
train_accuracy3 = np.zeros([201, 1], np.float32)
train_accuracy4 = np.zeros([201, 1], np.float32)

with tf.Session() as sess1:
  sess1.run(tf.global_variables_initializer())
  for i in range(20001):
    # batch = mnist.train.next_batch(50)
    ((X,Y_), data) = next_batch(data)
    if i % 100 == 0:
      train_accuracy1[int(i/100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy1[int(i/100)]))
    train_step1.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

  (tX,tY_) = test_data(data)
  print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))

with tf.Session() as sess2:
  sess2.run(tf.global_variables_initializer())
  for i in range(20001):
    # batch = mnist.train.next_batch(50)
    ((X,Y_), data) = next_batch(data)
    if i % 100 == 0:
      train_accuracy2[int(i/100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy2[int(i/100)]))
    train_step2.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

  (tX,tY_) = test_data(data)
  print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))

with tf.Session() as sess3:
  sess3.run(tf.global_variables_initializer())
  for i in range(20001):
    # batch = mnist.train.next_batch(50)
    ((X,Y_), data) = next_batch(data)
    if i % 100 == 0:
      train_accuracy3[int(i/100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy3[int(i/100)]))
    train_step3.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

  (tX,tY_) = test_data(data)
  print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))

with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer())
    for i in range(20001):
        # batch = mnist.train.next_batch(50)
        ((X, Y_), data) = next_batch(data)
        if i % 100 == 0:
            train_accuracy4[int(i / 100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy4[int(i / 100)]))
        train_step4.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

    (tX, tY_) = test_data(data)
    print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))

plt.figure()
plt.plot(train_accuracy, "k", label="AdamOpt, step-size = 1e-4")
plt.plot(train_accuracy1, "r", label="step-size = 1e-6")
plt.plot(train_accuracy2, "g", label="step-size = 5e-3")
plt.plot(train_accuracy3, "b", label="step-size = 1e-2")
plt.plot(train_accuracy4, "m", label="step-size = 9e-1")
plt.legend()
plt.show()
'''



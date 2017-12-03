import numpy as np
import os
import tensorflow as tf
import sys
import code
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'


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


def weight_variable(name, shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(name, initializer=initial)

def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def showImg(x):
  img = x.eval()
  for i in range(img.shape[3]):
    ax = plt.subplot(4, 8, i+1)
    ax.imshow(img[0,:,:,i])
  plt.show()


data = prep(50)


init = tf.constant(np.random.rand(1, 14, 14, 32), dtype=tf.float32)
x = tf.get_variable("x", initializer=init)
x_image = tf.reshape(x, [1, 14, 14, 32])

'''
# 1st convolutional layer
W_conv1 = weight_variable("W_conv1", [5, 5, 1, 32])
b_conv1 = bias_variable("b_conv1", [32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
'''

# 2nd convolutional layer
W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
b_conv2 = bias_variable("b_conv2", [64])

h_conv2 = tf.nn.relu(conv2d(x, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable("W_fc1", [7 * 7 * 64, 1024])
b_fc1 = bias_variable("b_fc1", [1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout
W_fc2 = weight_variable("W_fc2", [1024, 10])
b_fc2 = bias_variable("b_fc2", [10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




lam = 0.5
cls = 2

saver = tf.train.Saver(var_list=[W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2])
# training
optResp = lam * tf.reduce_sum(x * x) - tf.gather(tf.transpose(y_conv), cls)
train_step = tf.train.AdamOptimizer(1e-4).minimize(optResp, var_list=[x])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, "./model.ckpt")
  for i in range(100000):
    train_step.run(feed_dict={keep_prob: 0.5})

  showImg(x)
  #code.interact(local=locals())

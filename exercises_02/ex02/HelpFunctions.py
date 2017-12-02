from load import *
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

def training(y_, y_conv, trainMethod, data, x, keep_prob, jp):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = trainMethod.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = np.zeros([201, 1], np.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20001):
            if jp == 1:
                ((X, Y_), data) = next_batch(data)
            else:
                batch = mnist.train.next_batch(50)
                X = batch[0]
                Y_ = batch[1]

            if i % 100 == 0:
                train_accuracy[int(i / 100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy[int(i / 100)]))
            train_step.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

        if jp == 1:
            (tX, tY_) = test_data(data)
        else:
            tX = mnist.test.images
            tY_ = mnist.test.labels

        print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0}))

    return train_accuracy
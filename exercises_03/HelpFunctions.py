from load import *
import numpy as np
import tensorflow as tf
import code


def prep(k, x, y_, tx, ty_, org):
    n = x.shape[0] // k
    if org == 1:
        return (np.array_split(x,n), np.array_split(onehot(y_, 10),n), 0, n,
                tx, onehot(ty_, 10))
    else:
        return (np.array_split(x, n), np.array_split(y_, n), 0, n,
                tx, ty_)

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

def lrelu(a, x):
    return tf.maximum(x*a, x)


def training(y_, y_conv, trainMethod, data, x, keep_prob, iter, acc):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = trainMethod.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = np.zeros([int(np.ceil(iter/100)), 1], np.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter):
            ((X, Y_), data) = next_batch(data)
            if i % 100 == 0:
                train_accuracy[int(i / 100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy[int(i / 100)]))
                train_step.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5})

        (tX, tY_) = test_data(data)

        test_acc = accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0})
        print('test accuracy %g' % test_acc)
    if acc == 1:
        return train_accuracy, test_acc
    else:
        return 1-train_accuracy.mean(), 1-test_acc




def crossValidation(K, im, lab, y_, y_conv, x, keep_prob, iter):

    trainErr = np.zeros(K)
    testErr = np.zeros(K)

    for k in range(0, K):

        print('folds %g' % int(k+1))
        splitX = np.array_split(normalize(im), K)
        splitY = np.array_split((onehot(lab, 10)), K)
        testX = splitX[k]
        testY = splitY[k]
        del splitX[k]
        del splitY[k]
        trainX = np.concatenate(splitX)
        trainY = np.concatenate(splitY)

        dataCV = prep(50, trainX, trainY, testX, testY, 0)
        trainMethod = tf.train.AdamOptimizer(1e-3)

        (train_acc, test_acc) = training(y_, y_conv, trainMethod, dataCV, x, keep_prob, iter)
        testErr[k] = 1-test_acc
        trainErr[k] = (1 - train_acc).mean()

    avgTrainErr = trainErr.mean()
    avgTestErr = testErr.mean()
    return [avgTrainErr, avgTestErr]



def batch(x, phase, scope):
    with tf.variable_scope(scope):
        return tf.contrib.layers.batch_norm(x, center=True, scale=True,
                is_training=phase, scope='bn')

def batchAnonymous(x, phase):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True,
            is_training=phase)

def convLayer(shape, alpha, phase, useMaxPool, x):
    W_conv = weight_variable(shape)
    b_conv = bias_variable([shape[3]])
    h_conv = lrelu(alpha, conv2d(x, W_conv) + b_conv)
    h_norm = batchAnonymous(h_conv, phase)
    if useMaxPool:
        return max_pool_2x2(h_norm)
    else:
        return h_norm;


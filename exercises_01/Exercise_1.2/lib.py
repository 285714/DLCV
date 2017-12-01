import numpy as np
import os
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import sys
import code
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from scipy.misc import imread, imresize, imsave



# build DesignMatrix for single image of 3 channels
def prep(original, vignetted, n):

    # calculate point wise distance matrix
    # same as the example
    imgIn = imread(original)
    imgGoal = imread(vignetted)
    W = imgIn.shape[1]
    H = imgIn.shape[0]
    wc = W/2
    hc = H/2
    xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)
    r = np.sqrt(xv**2 + yv**2) / np.sqrt(wc**2 + hc**2)

    # take all 3 channels. Though the same factor is applied to each channel,
    # we might gain information that was lost as rounding error.
    RedMat = np.repeat(np.reshape(imgIn[:, :, 0], (-1, 1)), n + 1, axis=1)
    GreenMat = np.repeat(np.reshape(imgIn[:, :, 1], (-1, 1)), n + 1, axis=1)
    BlueMat = np.repeat(np.reshape(imgIn[:, :, 2], (-1, 1)), n + 1, axis=1)

    # reform the hyperparameter n into a matrix of the same size as the image
    length = imgIn.shape[0] * imgIn.shape[1]
    nMat = np.repeat(np.expand_dims(np.arange(n+1), axis=0), length, axis=0)

    # reshape the distance matrix so that we can do point wise multiplication
    DistMat = np.repeat(np.reshape(r, (-1, 1)), n+1, axis=1)
    hyperDistMat = DistMat ** nMat
    Red = RedMat * hyperDistMat
    Blue = BlueMat * hyperDistMat
    Green = GreenMat * hyperDistMat

    # form the entry matrix for a single image
    entryMat = np.concatenate((Red, Green, Blue), axis=0)

    # prepare the ground truth
    # transform the target image into column
    redGoal = np.reshape(imgGoal[:, :, 0], (-1, 1))
    greenGoal = np.reshape(imgGoal[:, :, 1], (-1, 1))
    blueGoal = np.reshape(imgGoal[:, :, 2], (-1, 1))
    Goal = np.concatenate((redGoal, greenGoal, blueGoal), axis=0)

    return [entryMat, Goal]



def prepall(n):

    [X1, Y1] = prep('cat_01.jpg', 'cat_01_vignetted.jpg', n)
    [X2, Y2] = prep('cat_01.jpg', 'cat_01_vignetted.jpg', n)
    [X3, Y3] = prep('cat_02.jpg', 'cat_02_vignetted.jpg', n)

    X = np.concatenate((X1, X2, X3), axis=0)
    Y = np.concatenate((Y1, Y2, Y3), axis=0)

    return [X,Y]



# train the model until convergence, returns trained parameter
def train(X, Y, n):
    # build the graph in TensorFlow
    a = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0))
    DesignMatrix = tf.placeholder(tf.float32, [None, n+1])
    # YPred = tf.matmul(tf.constant(X, dtype=tf.float32), a)
    YPred = tf.matmul(DesignMatrix, a)

    loss = tf.losses.mean_squared_error(Y, YPred)
    opt = tf.train.AdamOptimizer(0.01)
    train = opt.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train until convergence. One random inital point is sufficient to find
    # the global maximum since the model is convex.

    I = 10000
    eps = 0.01
    old = float('inf')
    for i in range(0, I):
        # results = sess.run([train, loss])
        results = sess.run([train, loss], feed_dict={DesignMatrix: X})
        sys.stdout.write("%d: %.5f      \r" % (i,results[1]) )
        sys.stdout.flush()
        if abs(results[1] - old) < eps:
            break
        else:
            old = results[1]

    return a.eval(sess)



def crossValidation(K, X, Y, trainFn, errorFn):

    trainErr = np.zeros(K)
    testErr = np.zeros(K)

    for k in range(0, K):

        splitX = np.array_split(X, 5)
        splitY = np.array_split(Y, 5)

        testX = splitX[k]
        testY = splitY[k]

        del splitX[k]
        del splitY[k]
        trainX = np.concatenate(splitX)
        trainY = np.concatenate(splitY)

        a = trainFn(trainX, trainY)
        trainErr[k] = errorFn(trainX, trainY, a)
        testErr[k] = errorFn(testX, testY, a)

    avgTrainErr = trainErr.mean()
    avgTestErr = testErr.mean()
    return [avgTrainErr, avgTestErr]

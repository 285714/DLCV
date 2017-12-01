import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from scipy.misc import imread, imresize



# Exercise 1.2


# build DesignMatrix for single image of 3 channels
def prepImg(original, vignetted, expo):

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

    # take all 3 channels
    # reshape images in each single color channel into columns
    # then repeat (n+1) times
    RedMat = np.repeat(np.reshape(imgIn[:, :, 0], (-1, 1)), expo + 1, axis=1)
    GreenMat = np.repeat(np.reshape(imgIn[:, :, 1], (-1, 1)), expo + 1, axis=1)
    BlueMat = np.repeat(np.reshape(imgIn[:, :, 2], (-1, 1)), expo + 1, axis=1)

    # reform the hyperparameter n into a matrix of the same size as the image
    length = imgIn.shape[0] * imgIn.shape[1]
    nMat = np.repeat(np.expand_dims(np.arange(expo+1), axis=0), length, axis=0)

    # reshape the distance matrix so that we can do point wise multiplication
    DistMat = np.repeat(np.reshape(r, (-1, 1)), expo+1, axis=1)
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


# Ex 1.2a , Ex 1.2b
# n is the hyperparameter
# 5-fold Cross-Validation to find optimal n


N = 20  # hyperspace
K = 5  # k-fold Cross-Validation
minErr = 1e+10
for n in range(1, N):

    [X1, Y1] = prepImg('cat_01.jpg', 'cat_01_vignetted.jpg', n)
    [X2, Y2] = prepImg('cat_01.jpg', 'cat_01_vignetted.jpg', n)
    [X3, Y3] = prepImg('cat_02.jpg', 'cat_02_vignetted.jpg', n)

    X = np.concatenate((X1, X2, X3), axis=0)
    Y = np.concatenate((Y1, Y2, Y3), axis=0)

    vSetNum = np.int(np.floor((Y.shape[0]) / 5))  # total number of the rows in a single validation set

    for k in range(1, K):
        # build the graph in TensorFlow
        a = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0))
        DesignMatrix = tf.placeholder(tf.float32, [None, n+1])
        YPred = tf.matmul(DesignMatrix, a)

        # prepare the training sets and validation sets
        head = (k-1)*vSetNum
        tail = (k*vSetNum) -1
        Yvalidation = Y[head:tail]
        Xvalidation = X[head:tail, :]
        Ytrain = np.delete(Y, range(head, tail), axis = 0)
        Xtrain = np.delete(X, range(head, tail), axis = 0)

        loss = tf.losses.mean_squared_error(Ytrain, YPred)
        opt = tf.train.AdamOptimizer(0.01)
        train = opt.minimize(loss)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        trainErr = sess.run(loss, feed_dict={DesignMatrix: Xtrain})

        I = 1000
        for i in range(0, I):
            results = sess.run([train, loss], feed_dict={DesignMatrix: Xtrain})
            print(i, I, results[1])

        aCandidate = a.eval(sess)

        # check performance of the trained optimal parameters on validation set
        YPred2 = tf.matmul(DesignMatrix, aCandidate)
        loss2 = tf.losses.mean_squared_error(Yvalidation, YPred2)
        trainErr = sess.run(loss2, feed_dict={DesignMatrix: Xvalidation})
        print(aCandidate, trainErr)

        x = np.arange(0, 1, 0.01)
        tmp = np.repeat(np.expand_dims(x, axis=1), n+1, axis=1) **\
            np.repeat(np.expand_dims(np.arange(n+1), axis=0), x.shape[0], axis=0)
        y = np.matmul(tmp, aCandidate)
        plt.plot(x, y)
        plt.show(block=False)

        # record the minimal train error and corresponding n and parameters a
        if trainErr < minErr:
            minErr = trainErr
            aOptimal = aCandidate
            nOPtimal = n

print(n, aOptimal, minErr)
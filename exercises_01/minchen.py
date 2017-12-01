import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import code

# prepare the inputs
# define function to calculate distance and image size
def computeDistance(I):

    # compute factor on complete grid
    W = I.shape[1]
    H = I.shape[0]
    wc = W / 2
    hc = H / 2

    xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)

    r = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)
    return [r, W, H]

# read in clean images as input
inputImg1 = imread('cat_01.jpg')
# inputImg2 = imread('cat_02.jpg')
# inputImg3 = imread('cat_03.jpg')
# read in vignetted images as goals
desireImg1 = imread('cat_01_vignetted.jpg')
# desireImg2 = imread('cat_02_vignetted.jpg')
# desireImg3 = imread('cat_03_vignetted.jpg')
# assume that all color channels are the same,
# so just take the second channel
inputImg1 = inputImg1[:, :, 0]
# inputImg2 = inputImg2[:, :, 1]
# inputImg3 = inputImg3[:, :, 1]
# desireImg1 = desireImg1[:, :, 1]
# desireImg2 = desireImg2[:, :, 1]
# desireImg3 = desireImg3[:, :, 1]

# compute distance, width and height for each image
DWH1 = computeDistance(inputImg1)
# DWH2 = computeDistance(inputImg2)
# DWH3 = computeDistance(inputImg3)

# Design Matrix:
# for each image, (X*r^i) in the
# polynomial function become
# columns in the design matrix;
# transform the matrices into column vectors
# and concatenate them together to form
# the design matrix.
X1_1 = np.reshape(inputImg1, (DWH1[1]*DWH1[2], 1))
X1_2 = np.reshape(inputImg1 * DWH1[0], (DWH1[1]*DWH1[2], 1))
X1_3 = np.reshape(inputImg1 * (DWH1[0] ** 2), (DWH1[1]*DWH1[2], 1))
X1_4 = np.reshape(inputImg1 * (DWH1[0] ** 3), (DWH1[1]*DWH1[2], 1))
X1_5 = np.reshape(inputImg1 * (DWH1[0] ** 4), (DWH1[1]*DWH1[2], 1))
X1_6 = np.reshape(inputImg1 * (DWH1[0] ** 5), (DWH1[1]*DWH1[2], 1))
X1_7 = np.reshape(inputImg1 * (DWH1[0] ** 6), (DWH1[1]*DWH1[2], 1))

# X2_1 = np.reshape(inputImg2, (DWH2[1]*DWH2[2], 1))
# X2_2 = np.reshape(inputImg2 * DWH2[0], (DWH2[1]*DWH2[2], 1))
# X2_3 = np.reshape(inputImg2 * (DWH2[0] ** 2), (DWH2[1]*DWH2[2], 1))
# X2_4 = np.reshape(inputImg2 * (DWH2[0] ** 3), (DWH2[1]*DWH2[2], 1))
# X2_5 = np.reshape(inputImg2 * (DWH2[0] ** 4), (DWH2[1]*DWH2[2], 1))
# X2_6 = np.reshape(inputImg2 * (DWH2[0] ** 5), (DWH2[1]*DWH2[2], 1))
# X2_7 = np.reshape(inputImg2 * (DWH2[0] ** 6), (DWH2[1]*DWH2[2], 1))

# X3_1 = np.reshape(inputImg3, (DWH3[1]*DWH3[2], 1))
# X3_2 = np.reshape(inputImg3 * DWH3[0], (DWH3[1]*DWH3[2], 1))
# X3_3 = np.reshape(inputImg3 * (DWH3[0] ** 2), (DWH3[1]*DWH3[2], 1))
# X3_4 = np.reshape(inputImg3 * (DWH3[0] ** 3), (DWH3[1]*DWH3[2], 1))
# X3_5 = np.reshape(inputImg3 * (DWH3[0] ** 4), (DWH3[1]*DWH3[2], 1))
# X3_6 = np.reshape(inputImg3 * (DWH3[0] ** 5), (DWH3[1]*DWH3[2], 1))
# X3_7 = np.reshape(inputImg3 * (DWH3[0] ** 6), (DWH3[1]*DWH3[2], 1))

# combine columns into matrices
X1 = np.concatenate((X1_1, X1_2, X1_3, X1_4, X1_5, X1_6, X1_7), axis=1)
# X2 = np.concatenate((X2_1, X2_2, X2_3, X2_4, X2_5, X2_6, X2_7), axis=1)
# X3 = np.concatenate((X3_1, X3_2, X3_3, X3_4, X3_5, X3_6, X3_7), axis=1)

code.interact(local=locals())

X = np.concatenate((X1, X2, X3), axis=0)

# form the vector of the goal pixel values
Y1 = np.reshape(desireImg1, (DWH1[1]*DWH1[2], 1))
Y2 = np.reshape(desireImg2, (DWH2[1]*DWH2[2], 1))
Y3 = np.reshape(desireImg3, (DWH3[1]*DWH3[2], 1))
Y = np.concatenate((Y1, Y2, Y3), axis=0)


# set up Design Matrix, Desired Predictions and other Inputs for TensorFlow

# randomly assign values as initial weights
a = tf.Variable(tf.random_uniform([7, 1], -1.0, 1.0))


DesignMatrix = tf.placeholder(tf.float32, [None, 7])
Weights = tf.placeholder(tf.float32, [7, 1])

YPred = tf.matmul(DesignMatrix, Weights)

loss = tf.losses.mean_squared_error(Y, YPred)
opt = tf.train.AdamOptimizer(0.01)
train = opt.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

results = sess.run([train, loss], feed_dict={DesignMatrix: X, Weights: sess.run(a)})
print(results[1])



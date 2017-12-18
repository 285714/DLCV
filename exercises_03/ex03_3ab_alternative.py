import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
from HelpFunctions import *

# prepare data
(im, lab, t_im, t_lab) = load_sample_dataset()
data = prep(50, im, lab, t_im, t_lab, 1)

x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y_ = tf.placeholder(tf.int64, shape=[None, 10])
# y_onehot = tf.one_hot(y_, 10)

x_image = tf.reshape(x, [-1, 28, 28, 1])


# Ex 3 (a) & (b)
# Implementation of Leaky ReLU and
# Playing around with different alpha with 5-folds Cross-Validation


# The hyper-parameters like the patch size (3x3) and number of feature maps(32 and 64)
# are defined according to the exercises 2.(f)

# Define the alpha for Leaky ReLU
# Leaky ReLU
# LReLU := x if x > 0
#          ax otherwise
#          a >= 0 and a <= 1
alphas1 = np.array([0,0.001,0.002,0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])
alphas2 = alphas1[1:10] * 10
alphas3 = alphas2 * 10
alphas = np.concatenate((alphas1,alphas2,alphas3))

iter = 20001
trainMethod = tf.train.AdamOptimizer(1e-3)
train_err = np.zeros(alphas.shape[0])
test_err = np.zeros(alphas.shape[0])

for n in range (alphas.shape[0]):

    a = alphas[n]
    # 1st convolutional layer
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    # Leaky ReLU layer 1
    h_conv1 = tf.maximum((conv2d(x_image, W_conv1) + b_conv1) * a, conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convolutional layer
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    # Leaky ReLU layer 2
    h_conv2 = tf.maximum((conv2d(h_pool1, W_conv2) + b_conv2) * a, conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # readout
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    [train_err[n], test_err[n]] = training(y_, y_conv, trainMethod, data, x, keep_prob, iter, 0)

fig = plt.figure(figsize=(19,10), frameon = False )
new_ticks = range(0, alphas.shape[0])
plt.plot(new_ticks, train_err, label="train error")
plt.plot(new_ticks, test_err, 'r--', label="test error")
lables = ['0','1e-3','2e-3','3e-3','4e-3','5e-3','6e-3','7e-3','8e-3','9e-3',
            '0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09',
            '0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
plt.xticks(new_ticks, lables)
plt.legend()
plt.xlabel("alpha for Leaky ReLU")
plt.show()


from __future__ import print_function
import config
import load, os, random
import numpy as np
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# batch_size = 128
nb_classes = 10
# nb_epoch = 12
nb_filters = 32
# size of pooling area for max pooling
# pool_size = (2, 2)
# convolution kernel size
# kernel_size = (3, 3)
img_rows, img_cols = config.img_row, config.img_col
img_path = config.img_path
X_train, y_train, X_test, y_test = load.load_sample_dataset()
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the images such that each image has mean zero and unit length
X_Train = np.zeros([X_train.shape[0], img_rows * img_cols], np.float32)
X_Test = np.zeros([X_test.shape[0], img_rows * img_cols], np.float32)

for n in range(0, X_train.shape[0]):
    X_Train[n, :] = np.reshape(X_train[n] - np.mean(X_train[n]), (1, -1))
    X_Train[n, :] = X_Train[n, :] / np.linalg.norm(X_train[n])

for m in range(0, X_test.shape[0]):
    X_Test[m, :] = np.reshape(X_test[m] - np.mean(X_test[m]), (1, -1))
    X_Test[m, :] = X_Test[m,:] / np.linalg.norm(X_test[m])

# print(np.argwhere(np.isnan(X_Test)))


# convert class vectors to one-hot

Y_train = tf.one_hot(y_train, nb_classes)
Y_test = tf.one_hot(y_test, nb_classes)

# build up model
x = tf.placeholder(tf.float32, [None, img_rows * img_cols])
W = tf.Variable(tf.zeros([img_rows * img_cols, nb_classes]))
b = tf.Variable(tf.zeros([nb_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(10000):
    sess.run(train_step, feed_dict={x: X_Train, y_: np.squeeze(sess.run(Y_train))})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X_Test, y_: np.squeeze(sess.run(Y_test))}))
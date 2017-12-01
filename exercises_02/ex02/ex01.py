from __future__ import print_function
import config
import load
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# (a) & (b)
nb_classes = 10

img_rows, img_cols = config.img_row, config.img_col

X_train, y_train, X_test, y_test = load.load_sample_dataset()

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

# check whether there are some NaN
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

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    sess.run(train_step, feed_dict={x: X_Train, y_: np.squeeze(sess.run(Y_train))})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X_Test, y_: np.squeeze(sess.run(Y_test))}))


# (c) Visualize the weights vector and the corresponding examples
'''
trained_Weights = sess.run(W)
C = np.zeros([10,10],int)
for c in range(10):
    C[c,:] = np.where(y_train == c)[0][0:10]


Weights_image = np.reshape(trained_Weights, [28,28,10])
for img in range(10):
    plt.subplot(11, 10, img+1)
    plt.imshow(Weights_image[:,:,img])
    for exp in range(10):
        for exp2 in range(10):
            plt.subplot(11, 10, (exp+1)*10+(exp2+1))
            plt.imshow(X_train[C[exp2,exp],:,:])

plt.show()
'''
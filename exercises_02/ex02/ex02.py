import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
from HelpFunctions import *


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

# (d) remove the convolutional layers

'''
trainMethod = tf.train.AdamOptimizer(1e-4)

# remove the 2nd conv layer
W_fc_only1 = weight_variable([14 * 14 * 32, 1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1_only1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc_only1) + b_fc1)
h_fc1_drop_only1 = tf.nn.dropout(h_fc1_only1, keep_prob)
y_conv_only1 = tf.matmul(h_fc1_drop_only1, W_fc2) + b_fc2
train_accuracy_only1 = training(y_, y_conv_only1, trainMethod, data, x, keep_prob, 1)

# remove the 1st conv layer
# by setting the number of feature maps of the 1st conv layer to 64
W_conv = weight_variable([5, 5, 1, 64])
h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv2)
h_pool = max_pool_2x2(h_conv)
W_fc = weight_variable([14 * 14 * 64, 1024])
h_pool_flat = tf.reshape(h_pool, [-1, 14*14*64])
h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc1)
h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
y_conv_only2 = tf.matmul(h_fc_drop, W_fc2) + b_fc2
train_accuracy_only2 = training(y_, y_conv_only2, trainMethod, data, x, keep_prob, 1)


plt.figure()
plt.plot(train_accuracy, "k", label="2 convolutional layers")
plt.plot(train_accuracy_only2, "g", label="removed the 1st convolutional layer")
plt.plot(train_accuracy_only1, "r", label="removed the 2nd convolutional layer")
plt.legend()
plt.show()
'''

# (e) Try different step sizes for Gradient Descent Optimizer

'''
train_step1 = tf.train.GradientDescentOptimizer(1e-6)
train_step2 = tf.train.GradientDescentOptimizer(5e-3)
train_step3 = tf.train.GradientDescentOptimizer(1e-2)
train_step4 = tf.train.GradientDescentOptimizer(9e-1)

train_accuracy_1 = training(y_, y_conv, train_step1, data, x, keep_prob, 1)
train_accuracy_2 = training(y_, y_conv, train_step2, data, x, keep_prob, 1)
train_accuracy_3 = training(y_, y_conv, train_step3, data, x, keep_prob, 1)
train_accuracy_4 = training(y_, y_conv, train_step4, data, x, keep_prob, 1)


plt.figure()
plt.plot(train_accuracy, "k", label="AdamOpt, step-size = 1e-4")
plt.plot(train_accuracy_1, "r", label="step-size = 1e-6")
plt.plot(train_accuracy_2, "g", label="step-size = 5e-3")
plt.plot(train_accuracy_3, "b", label="step-size = 1e-2")
plt.plot(train_accuracy_4, "m", label="step-size = 9e-1")
plt.legend()
plt.show()
'''



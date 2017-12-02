from ex02 import *


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int64, shape=[None, 10])
# y_onehot = tf.one_hot(y_, 10)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# (1) copy over the trained weights and biases from ex02
#     only train the classifier

# 1st convolutional layer
W_conv1_jp = tf.Variable(W1, trainable=False)
b_conv1_jp = tf.Variable(b1, trainable=False)

h_conv1_jp = tf.nn.relu(conv2d(x_image, W_conv1_jp) + b_conv1_jp)
h_pool1_jp = max_pool_2x2(h_conv1_jp)

# 2nd convolutional layer
W_conv2_jp = tf.Variable(W2, trainable=False)
b_conv2_jp = tf.Variable(b2, trainable=False)

h_conv2_jp = tf.nn.relu(conv2d(h_pool1_jp, W_conv2_jp) + b_conv2_jp)
h_pool2_jp = max_pool_2x2(h_conv2_jp)

# densely connected layer
W_fc1_jp = tf.Variable(Wfc, trainable=False)
b_fc1_jp = tf.Variable(bfc, trainable=False)

h_pool2_flat_jp = tf.reshape(h_pool2_jp, [-1, 7*7*64])
h_fc1_jp = tf.nn.relu(tf.matmul(h_pool2_flat_jp, W_fc1_jp) + b_fc1_jp)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop_jp = tf.nn.dropout(h_fc1_jp, keep_prob)

# readout
W_fc2_jp = weight_variable([1024, 10])
b_fc2_jp = bias_variable([10])

y_conv_jp = tf.matmul(h_fc1_drop_jp, W_fc2_jp) + b_fc2_jp


# training
trainMethod = tf.train.AdamOptimizer(1e-4)
train_accuracy = training(y_, y_conv_jp, trainMethod, data, x, keep_prob, 0)


# (2) Comparison

# test accuracies:
# with trained weights and biases for jp :  96.95%
# with all weights and biases trained:      99.24%
# with logits regression:                   91.87%


'''
# train all the weights and biases for handwritten digits with covnet
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
train_accuracy_digits = training(y_, y_conv, trainMethod, data, x, keep_prob, 0)


# train with logistic regression

# x = tf.placeholder(tf.float32, [None, 784])
# y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy_lr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step_lr = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_lr)
correct_prediction_lr = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy_lr = tf.reduce_mean(tf.cast(correct_prediction_lr, tf.float32))
train_accuracy_lr = np.zeros([201, 1], np.float32)
with tf.Session() as sess_lr:
  sess_lr.run(tf.global_variables_initializer())
  for i in range(20001):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy_lr[int(i/100)] = accuracy_lr.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy_lr[int(i/100)]))
    train_step_lr.run(feed_dict={x: batch[0], y_: batch[1]})

  print('test accuracy %g' % accuracy_lr.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))


plt.figure()
plt.plot(train_accuracy, "k", label="with trained weights and biases for jp ")
plt.plot(train_accuracy_digits, "r", label="all weights and biases trained")
plt.plot(train_accuracy_lr, "g", label="with logits regression")
plt.legend()
plt.show()

'''

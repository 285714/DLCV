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
phase = tf.placeholder(tf.bool, name='phase')

x_image = tf.reshape(x, [-1, 28, 28, 1])



# Ex 3 (d)

alpha = tf.get_variable("alpha",
                        shape = 1,
                        initializer = tf.constant_initializer(0.0),
                        dtype = tf.float32)

# 1st convolutional layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = lrelu(alpha, conv2d(x_image, W_conv1) + b_conv1)
h_norm1 = batch(h_conv1, phase, 'layer1')
h_pool1 = max_pool_2x2(h_norm1)

# 2nd convolutional layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = lrelu(alpha, conv2d(h_pool1, W_conv2) + b_conv2)
h_norm2 = batch(h_conv2, phase, 'layer2')
h_pool2 = max_pool_2x2(h_norm2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = lrelu(alpha, batch(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, phase, 'layer3'))
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
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
            train_accuracy[int(i/100)] = accuracy.eval(feed_dict={x: X, y_: Y_, keep_prob: 1.0, phase: 0})
            print('step %d, training accuracy %g' % (i, train_accuracy[int(i/100)]))
        train_step.run(feed_dict={x: X, y_: Y_, keep_prob: 0.5, phase: 1})

    (tX,tY_) = test_data(data)
    print('test accuracy %g' % accuracy.eval(feed_dict={x: tX, y_: tY_, keep_prob: 1.0, phase: 0}))
    print(sess.run(alpha))

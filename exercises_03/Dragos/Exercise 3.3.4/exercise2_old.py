# -*- coding: utf-8 -*-
"""
@authors: Grama Dragos, Alexander Scholl and Erik Da

"""

#####                                   EXEERCISE 2                               ######

#2.1

# Import the dataset
import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
filename = 'train_test_file_list.h5'
f = h5py.File(filename, 'r')

# Get the data
print("Keys: %s" % f.keys())
a0_group_key = list(f.keys())[0]
a1_group_key = list(f.keys())[1]
a2_group_key = list(f.keys())[2]
a3_group_key = list(f.keys())[3]

train_x = list(f[a2_group_key])
train_y = list(f[a3_group_key])
test_x = list(f[a0_group_key])
test_y = list(f[a1_group_key])


# Transform the data
reshape_train_x=np.zeros((19909,784))
reshape_train_y=np.zeros((19909,10))
reshape_test_x=np.zeros((3514,784))
reshape_test_y=np.zeros((3514,10))
for k in range(0,19909):
    for i in range(0,28):
        for j in range(0,28):
            reshape_train_x[k][i*28+j]=train_x[k][i][j]
for k in range(0,3514):
    for i in range(0,28):
        for j in range(0,28):
            reshape_test_x[k][i*28+j]=test_x[k][i][j]      

for k in range(0,19909): 
         reshape_train_y[k,train_y[k][0]]=1      
         reshape_train_x[k]=reshape_train_x[k]-(sum(reshape_train_x[k])/784) 
         reshape_train_x[k]=reshape_train_x[k]/np.std(reshape_train_x[k])
    

for k in range(0,3514):   
         reshape_test_y[k,test_y[k][0]]=1    
         reshape_test_x[k]=reshape_test_x[k]-(sum(reshape_test_x[k])/784) 
         reshape_test_x[k]=reshape_test_x[k]/np.std(reshape_test_x[k])   
    
train_x=reshape_train_x    
train_y=reshape_train_y
test_x=reshape_test_x 
test_y=reshape_test_y 
    

# Build the model

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
model = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-entropy loss function

y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))

# Training

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    rows=np.random.randint(0,train_x.shape[0],5000)   
    batch_xs = train_x[rows]
    batch_ys = train_y[rows]
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    
# See the accuracy of the model

acc=sess.run(accuracy, feed_dict={x: train_x, y: train_y})
print("Accuracy on train-set: {0:.2%}".format(acc))

acc=sess.run(accuracy, feed_dict={x: test_x, y: test_y})
print("Accuracy on test-set: {0:.2%}".format(acc))



# 2.2

def new_weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def new_biases(lenght):
  initial = tf.constant(0.05, shape=lenght)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
  
  
# First Convolutional Layer
W_conv1 = new_weights([5, 5, 1, 32])
b_conv1 = new_biases([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second Convolutional Layer

W_conv2 = new_weights([5, 5, 32, 64])
b_conv2 = new_biases([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Conected Layer

W_fc1 = new_weights([7 * 7 * 64, 1024])
b_fc1 = new_biases([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Final Layer

W_fc2 = new_weights([1024, 10])
b_fc2 = new_biases([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Training the model
def training_the_model(step_size):
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(step_size).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    x_axis=np.zeros(50)
    y_axis=np.zeros(50)
    k=0
    plt.axis([0,5000,0,1])
    tf.global_variables_initializer().run()
    for i in range(5000):
        rows=np.random.randint(0,train_x.shape[0],50)   
        batch_xs = train_x[rows]
        batch_ys = train_y[rows]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                            x: batch_xs, y: batch_ys, keep_prob: 1.0})
            x_axis[k]=i
            y_axis[k]=train_accuracy
            k=k+1
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
    plt.plot(x_axis, y_axis, 'ro')


#
training_the_model(1e-4)

    
""" 
 When you remove one layer the model is faster. On training data you get the same accuracy
 (100%) but on the test data you get a lower accuracy than before removing the layer. In 
 conclusion removing one layer makes the model run faster and get a higher accuracy faster
 for the training data but the model doesn't perform on the test data as good as the model with
 2 convolutional layers.
 
 """

# Training the model with very high step-size 
training_the_model(1)  

# Training the model with very low step-size
training_the_model(1e-9)







from a import *
import math


def ceildiv(n, k):
    return math.ceil(n / k)

print('c:', c, '- x1:', x1, '- x2:', x2, '- x3:', x3, '- x4:', x4, '- x5:', x5, '- x6:', x6, '- x7:', x7, '- x8:', x8)

W2_d = tf.Variable(tf.zeros([3,3,x5,x6]))
layer2_d = tf.nn.conv2d_transpose(layer2, W2_d,
        output_shape=layer1.shape, strides=[1, x7, x8, 1], padding='SAME')

W1_d = tf.Variable(tf.zeros([3,3,x1,x2]))
layer1_d = tf.nn.conv2d_transpose(layer2_d, W1_d,
        output_shape=x.shape, strides=[1, x3, x4, 1], padding='SAME')


code.interact(local=locals())

import tensorflow as tf
import numpy as np
import code
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def rand_subset(arr):
    return np.random.choice(arr, random.randint(0,len(arr)), replace=False)


# ceiling 64 / 6 = 11
# ceiling 48 / [12-15] = 4

c  = random.randint(12,15)
x1 = 17
x2 = random.randint(1,1000)
x3 = np.prod(rand_subset(prime_factors(6)))
x4 = np.prod(rand_subset(prime_factors(c)))
x5 = x2
x6 = 13
x7 = 6 / x3
x8 = c / x4

size = 1000
x = tf.placeholder(np.float32, [size,64,48,17])
W1 = tf.Variable(tf.zeros([3,3,x1,x2]))
layer1 = tf.nn.conv2d(x, W1, strides=[1, x3, x4, 1], padding='SAME')
W2 = tf.Variable(tf.zeros([3,3,x5,x6]))
layer2 = tf.nn.conv2d(layer1, W2, strides=[1, x7, x8, 1], padding='SAME')

# code.interact(local=locals())

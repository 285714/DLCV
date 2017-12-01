import numpy as np
import tensorflow as tf

import code
import sys
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


n = 6

def prepImg( original, vignetted ):
    img = imread( original )
    img2 = imread( vignetted )

    W = img.shape[1]
    H = img.shape[0]
    wc = W/2
    hc = H/2

    xv,yv = np.meshgrid( np.arange( W ) - wc, np.arange( H ) - hc )
    r = np.sqrt( xv**2 + yv**2 ) / np.sqrt( wc**2 + hc**2 )

    red = img[:,:,0]
    red2 = img2[:,:,0]

    y = np.reshape(red, (-1,1))
    Y = np.repeat(y, n+1, axis=1)
    length = y.shape[0]
    k = np.repeat(np.expand_dims(np.arange(n+1), axis=0), length, axis=0)
    R = np.repeat(np.reshape(r, (-1,1)), n+1, axis=1)
    X = R ** k

    return [X * Y, np.reshape(red2, (-1,1))]


a0 =  1.0
a1 = -0.3
a2 =  0.05
a3 = -0.25
a4 = -0.4
a5 = -0.05
a6 =  0.1


[X1, Y1] = prepImg('cat_01.jpg', 'cat_01_vignetted.jpg')
[X2, Y2] = prepImg('cat_01.jpg', 'cat_01_vignetted.jpg')
[X3, Y3] = prepImg('cat_02.jpg', 'cat_02_vignetted.jpg')

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

#  code.interact(local=locals())


a = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0)) # tf.Variable( [[a0],[a1],[a2],[a3],[a4],[a5],[a6]] )
DesignMatrix = tf.placeholder( tf.float32, shape=(Y.shape[0],n+1) )
YPred = tf.matmul(DesignMatrix, a)

loss = tf.losses.mean_squared_error(Y, YPred)
opt = tf.train.AdamOptimizer(0.01)
train = opt.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


I = 1000
for i in range(0,I):
    results = sess.run([train, loss], feed_dict={DesignMatrix: X})
    # print(results[1]))
    sys.stdout.write("%d/%d: %.3f   \r" % (i,I,results[1]) )
    sys.stdout.flush()

a_ = a.eval(sess)
print(a_)



x = np.arange(0,1,0.01)
tmp = np.repeat(np.expand_dims(x, axis=1), n+1, axis=1) **\
         np.repeat(np.expand_dims(np.arange(n+1), axis=0), x.shape[0], axis=0)
y = np.matmul(tmp, a_)

plt.plot(x,y)
plt.show()


code.interact(local=locals())



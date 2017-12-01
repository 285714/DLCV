import numpy as np
import tensorflow as tf
import code


theta = tf.Variable([2,1,0], name='x', dtype=tf.float32)
theta1, theta2, theta3 = tf.split(0, 3, theta)
e = theta1

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(e)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

results = sess.run([train,e])
print(results)



code.interact(local=locals())

theta = tf.Variable([2,1,0], dtype=tf.float32)
theta2, theta3 = tf.split(0, 3, theta)
e = tf.log(theta1) # 2 * (theta1 ** 2) + 4 * theta2 + tf.maximum(0, theta2 + theta3)

# loss = tf.losses.mean_squared_error(0, e)
opt = tf.train.GradientDescentOptimizer(0.5)
train = opt.minimize(e)

code.interact(local=locals())

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



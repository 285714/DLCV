from lib import *


n = 10


def train2(X, Y, l):
    # build the graph in TensorFlow
    a = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0))
    DesignMatrix = tf.placeholder(tf.float32, [None, n+1])
    YPred = tf.matmul(DesignMatrix, a)

    mse = tf.losses.mean_squared_error(Y, YPred)
    loss = mse + 0.5 * l * tf.reduce_sum(a ** 2)
    opt = tf.train.AdamOptimizer(0.001)
    train = opt.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train until convergence.
    I = 10000
    eps = 0.01
    old = float('inf')
    for i in range(0, I):
        results = sess.run([train, loss], feed_dict={DesignMatrix: X})

        if abs(results[1] - old) < eps:
            break
        else:
            old = results[1]

    return a.eval(sess)



L = 51
K = 5


[X,Y] = prepall(n)
avgTrainErrs = np.zeros(L)
avgTestErrs = np.zeros(L)
for l in range(0, L):
    def trainFn(trainX, trainY): return train2(trainX, trainY, l)
    def errorFn(X, Y, a): return ((np.matmul(X, a) - Y) ** 2).mean() # error function does not contain lambda..
    [avgTrainErrs[l], avgTestErrs[l]] = crossValidation(K, X, Y, trainFn, errorFn)
    print(l)


ns = np.arange(0,L)
plt.plot(ns, avgTrainErrs, label="train error")
plt.plot(ns, avgTestErrs, label="test error")
plt.legend()
plt.xlabel("lambda")
plt.show()


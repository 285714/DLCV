from lib import *


n = 7


def train2(X, Y, l):
    # build the graph in TensorFlow
    a = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0))
    DesignMatrix = tf.placeholder(tf.float32, [None, n+1])
    YPred = tf.matmul(tf.constant(X, dtype=tf.float32), a)

    mse = tf.losses.mean_squared_error(Y, YPred)
    loss = mse + 0.5 * l * tf.reduce_sum(a ** 2)
    opt = tf.train.AdamOptimizer(0.01)
    train = opt.minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # train until convergence.
    I = 10000
    eps = 0.01
    old = math.inf
    for i in range(0, I):
        results = sess.run([train, loss])
        sys.stdout.write("%d: %.5f      \r" % (i,results[1]) )
        sys.stdout.flush()
        if abs(results[1] - old) < eps:
            break
        else:
            old = results[1]

    return a.eval(sess)




L = 20
K = 5


[X,Y] = prepall(n)
avgTrainErrs = np.zeros(L)
avgTestErrs = np.zeros(L)
for l in range(0, L):
    def trainFn(trainX, trainY): return train2(trainX, trainY, l)
    def errorFn(X, Y, a): return ((np.matmul(X, a) - Y) ** 2).mean() # error function does not contain lambda..
    [avgTrainErrs[l], avgTestErrs[l]] = crossValidation(K, X, Y, trainFn, errorFn)



ns = np.arange(0,L)
plt.plot(ns, avgTrainErrs, label="train error")
plt.plot(ns, avgTestErrs, label="test error")
plt.legend()
plt.xlabel("lambda")
plt.show(block=False)

code.interact(local=locals())

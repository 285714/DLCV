from lib import *



N = 4
K = 5

avgTrainErrs = np.zeros(N)
avgTestErrs = np.zeros(N)
for n in range(0, N):
    [X,Y] = prepall(n)
    def trainFn(X, Y): return train(X, Y, n)
    def errorFn(X, Y, a): return ((np.matmul(X, a) - Y) ** 2).mean()
    [avgTrainErrs[n], avgTestErrs[n]] = crossValidation(K, X, Y, trainFn, errorFn)
    print(n)



ns = np.arange(0,N)
plt.plot(ns, avgTrainErrs, label="train error")
plt.plot(ns, avgTestErrs, 'r--', label="test error")
plt.legend()
plt.xlabel("n")
plt.show(block=False)

code.interact(local=locals())




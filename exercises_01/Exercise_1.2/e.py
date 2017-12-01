from lib import *
import matplotlib.mlab as mlab

[X1,Y1] = prep('cat_01.jpg','cat_01_devignetted.png',0)
[X2,Y2] = prep('cat_02.jpg','cat_02_devignetted.png',0)
[X3,Y3] = prep('cat_03.jpg','cat_03_devignetted.png',0)
X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

noise = np.squeeze(X - Y).astype(np.int64)
min = np.min(noise)
max = np.max(noise)
bins = np.bincount(noise - min)
length = noise.shape[0]

# Simply compute the sample variance on the data

mean = 0
sampleVariance = np.sum((noise - mean) ** 2) / length
sampleSigma = math.sqrt(sampleVariance)

x = np.arange(0,bins.shape[0]) + min
plt.plot(x, bins)

x = np.linspace(min, max, num=200)
plt.plot(x, length * mlab.normpdf(x, mean, sampleSigma))
plt.show(block=False)


print('estimated standard deviation = ', sampleSigma)


code.interact(local=locals())


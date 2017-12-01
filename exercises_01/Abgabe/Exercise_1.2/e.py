from lib import *
import matplotlib.mlab as mlab

# The model introduced for vignetting is not helpful to reduce the gaussian noise in the images,
# because the gaussian noise is an additive noise and has nothing to do with the pixel values and their
# distances to the image center, but only random variables that obey the normal distribution with specific
# mean and standard deviation.

# While the model was being trained with the training set, the gaussian noise was not removed,
# so every time the loss function was evaluated with the noisy target images, which made
# the model more likely to bring the pixel values of the clean images towards vignetting but not
# a "clean" vignetting. Even though the parameters were trained such that the mean square error was minimal,
# it is not guaranteed the input clean images could be mapped to the "clean" vignetted images.
# The noise could be calculated as :  noise = noisy_vignetted_image - clean_vignetted_image, but the
# "clean" vignetted images cannot be computed correctly 100% with the model.
# Further more, the additive noise could have the chance to bring the pixel values larger than 255 or
# smaller than 0, the real values of the noise could also be concealed by for example the forced
# data type conversion for imshow()

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

# Simply compute the sample variance on the dataS

mean = 0
sampleVariance = np.sum((noise - mean) ** 2) / length
sampleSigma = math.sqrt(sampleVariance)

x = np.arange(0,bins.shape[0]) + min
plt.plot(x, bins)

x = np.linspace(min, max, num=200)
plt.plot(x, length * mlab.normpdf(x, mean, sampleSigma))
plt.show()


print('estimated standard deviation = ', sampleSigma)


#code.interact(local=locals())


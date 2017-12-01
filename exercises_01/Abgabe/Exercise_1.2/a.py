from lib import *

# assume the optimal hyperparameter n = 7 (from 0 to 6)
n = 6

[X,Y] = prepall(n)
a = train(X, Y, n)

img1 = imread('cat_01.jpg')
img2 = imread('cat_02.jpg')
img3 = imread('cat_03.jpg')
img1V = imread('cat_01_vignetted.jpg')
img2V = imread('cat_02_vignetted.jpg')
img3V = imread('cat_03_vignetted.jpg')
# Show the original image
plt.subplot(3, 3, 1)
plt.imshow(img1)
plt.subplot(3, 3, 2)
plt.imshow(img2)
plt.subplot(3, 3, 3)
plt.imshow(img3)
# Show the original vignetted image
plt.subplot(3, 3, 4)
plt.imshow(img1V)
plt.subplot(3, 3, 5)
plt.imshow(img2V)
plt.subplot(3, 3, 6)
plt.imshow(img3V)
# Show the vignetted image
plt.subplot(3, 3, 7)
plt.imshow(np.uint8(vignetting( img1,a )))
plt.subplot(3, 3, 8)
plt.imshow(np.uint8(vignetting( img2,a )))
plt.subplot(3, 3, 9)
plt.imshow(np.uint8(vignetting( img3,a )))

plt.show()
#code.interact(local=locals())

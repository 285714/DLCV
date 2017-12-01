from lib import *


'''
from a import *
a = np.squeeze(a)
'''

a = np.array([0.96725887, -0.15805617, 0.42861691, -0.52531856, -0.97843331,-0.043388, -0.30630532, 0.81417495])



# the function is just the inverse of the formula given in the exercise per
# pixel. The values are clamped at the end, otherwise imsave will scale them.
def devignette(a, vignetted, original):
    n = a.shape[0] - 1

    imgVign = imread(vignetted)
    W = imgVign.shape[1]
    H = imgVign.shape[0]
    wc = W/2
    hc = H/2
    xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)
    r = np.sqrt(xv**2 + yv**2) / np.sqrt(wc**2 + hc**2)

    RedVign   = np.squeeze(np.reshape(imgVign[:, :, 0], (-1, 1)))
    GreenVign = np.squeeze(np.reshape(imgVign[:, :, 1], (-1, 1)))
    BlueVign  = np.squeeze(np.reshape(imgVign[:, :, 2], (-1, 1)))

    length = W * H
    nMat = np.repeat(np.expand_dims(np.arange(n+1), axis=0), length, axis=0)

    DistMat = np.repeat(np.reshape(r, (-1, 1)), n+1, axis=1)
    hyperDistMat = DistMat ** nMat
    polyMat = np.matmul(hyperDistMat, a)

    RedOrig   = RedVign / polyMat
    GreenOrig = GreenVign / polyMat
    BlueOrig  = BlueVign / polyMat

    imgOrig = np.transpose(np.array([np.reshape(RedOrig, (H,W)), \
            np.reshape(GreenOrig, (H,W)), np.reshape(BlueOrig, (H,W))]), (1,2,0))

    imsave(original, np.maximum(0, np.minimum(imgOrig, 255)))



for i in range(1, 7):
    devignette(a, 'cat_0' + str(i) + '_vignetted.jpg', 'cat_0' + str(i) + '_devignetted.png')


'''
n = 7
x = np.arange(0,1,0.01)
tmp = np.repeat(np.expand_dims(x, axis=1), n+1, axis=1) **\
         np.repeat(np.expand_dims(np.arange(n+1), axis=0), x.shape[0], axis=0)
y = np.matmul(tmp, a)

plt.plot(x,y)
plt.show()
'''


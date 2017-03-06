import cv2
import numpy as np

def loadImage(name):
    image = cv2.imread(name)
    h, w = image.shape[0], image.shape[1]
    t, b, s = 55, 15, 10
    image = image[t:(h-b),s:(w-s),:]
    return image

def augmentImage(image):
    h, w, c = image.shape
    [x1, x2] = np.random.choice(w, 2, replace=False)
    m = h / (x2 - x1)
    b = - m * x1
    for i in range(h):
        c = int((i - b) / m)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    #image = image * 0.5
    return image

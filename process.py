import cv2
import numpy as np

# Load image and crop image bottom and top and sides
def loadImage(name):
    image = cv2.imread(name)
    h, w = image.shape[0], image.shape[1]
    t, b, s = 55, 15, 10
    image = image[t:(h-b),s:(w-s),:]
    return image

# Add a random shadow to the image. This works by randomly generating a line
# that goes from bottom to top of screen and we cut the brightness of each
# pixel by half on left side of the shadow
# TODO: instead of always using the left side for the shadow, randomize which
# side gets the shadow
def augmentImage(image):
    h, w, c = image.shape
    [x1, x2] = np.random.choice(w, 2, replace=False)
    m = h / (x2 - x1)
    b = - m * x1
    for i in range(h):
        c = int((i - b) / m)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image

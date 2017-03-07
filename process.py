import cv2
import numpy as np

# Load image and crop image bottom and top and sides
def loadImage(name):
    image = cv2.imread(name)
    #h, w = image.shape[0], image.shape[1]
    #t, b, s = 55, 15, 10
    #image = image[t:(h-b),s:(w-s),:]
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

if __name__ == "__main__":
    names = [
        "track1",
        "track2",
        "track1-recovery1",
        "track1-recovery2",
        "track1-recovery3",
        "track2-recovery1",
        "track2-recovery2",
        "track2-recovery3"
    ]
    for name in names:
        image = loadImage("images/"+name+".jpg")
        image = augmentImage(image)
        image = image[:, ::-1, :]
        cv2.imwrite("images/"+name+"-processed.jpg", image)
        # crop
        h, w = image.shape[0], image.shape[1]
        t, b, s = 55, 15, 10
        image = image[t:(h-b),s:(w-s),:]
        cv2.imwrite("images/"+name+"-cropped.jpg", image)

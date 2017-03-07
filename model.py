import csv
import sys
import numpy as np
import cv2
import random
import tensorflow as tf
import matplotlib.pyplot as plt
tf.python.control_flow_ops = tf
from keras.layers import core, convolutional, pooling, Cropping2D, Lambda
from keras import models, optimizers, backend
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# external files
from preprocess import load_samples
import process

# generator used to load images and also augment data
# the generator takes an augment argument which will be true during training
# and false during testing and validation
#
# When augmenting we randomly select a center, left or right image and
# add a correction factor to the steeting
def generator(samples, path, augment=True, batch_size=128):
    num_samples = len(samples)
    correction = [0, 0.2, -0.2]

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            # load images and angles
            for batch_sample in batch_samples:
                # load the image (this step crops the image as well)
                #index = random.randint(0,2)
                name = path + '/IMG/'+batch_sample[0].split('/')[-1]
                image = process.loadImage(name)

                # augment the data by creating shadows
                #if augment:
                #    image = process.augmentImage(image)

                # normalize the image
                #image = image / 255.0 - 0.5

                # generate the angle and add correction
                angle = float(batch_sample[1])# + correction[index]

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            # randomly flip half of the images
            #batch_len = X_train.shape[0]
            #indices = random.sample(range(batch_len), int(batch_len/2.0))
            #X_train[indices] = X_train[indices, :, ::-1, :]
            #y_train[indices] = -y_train[indices]

            # send the batch to the trainer
            yield shuffle(X_train, y_train)


# load and split data into training set and validation set
#data_path = 'data'
data_path = '/home/rrodriguez/data'
samples = load_aug(data_path, 'driving_log_processed.csv')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, data_path, augment=True)
validation_generator = generator(validation_samples, data_path, augment=False)

model = models.Sequential()
model.add(Cropping2D(cropping=((55,15), (10,10)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(convolutional.Convolution2D(24, 5, 5, activation='relu', input_shape=(90,300,3)))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(36, 5, 5, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(48, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                 samples_per_epoch=len(train_samples),
                 validation_data=validation_generator,
                 nb_val_samples=len(validation_samples),
                 nb_epoch=30)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# was giving an error when exiting, this makes sure that all TF sessions are
# closed before attempting to exit
backend.clear_session()

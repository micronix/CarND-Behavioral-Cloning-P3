**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[track1]: ./images/track1.jpg "Center"
[track2]: ./images/track2.jpg "Center"
[recovery1]: ./images/track1-recovery1.jpg "Recovery"
[recovery2]: ./images/track1-recovery2.jpg "Recovery"
[recovery3]: ./images/track1-recovery3.jpg "Recovery"
[recovery4]: ./images/track2-recovery1.jpg "Recovery"
[recovery5]: ./images/track2-recovery2.jpg "Recovery"
[recovery6]: ./images/track2-recovery3.jpg "Recovery"
[track1-processed]: ./images/track1-processed.jpg "Center"
[track2-processed]: ./images/track2-processed.jpg "Center"
[recovery1-processed]: ./images/track1-recovery1-processed.jpg "Recovery"
[recovery2-processed]: ./images/track1-recovery2-processed.jpg "Recovery"
[recovery3-processed]: ./images/track1-recovery3-processed.jpg "Recovery"
[recovery4-processed]: ./images/track2-recovery1-processed.jpg "Recovery"
[recovery5-processed]: ./images/track2-recovery2-processed.jpg "Recovery"
[recovery6-processed]: ./images/track2-recovery3-processed.jpg "Recovery"
[track1-cropped]: ./images/track1-cropped.jpg "Center"
[track2-cropped]: ./images/track2-cropped.jpg "Center"
[recovery1-cropped]: ./images/track1-recovery1-cropped.jpg "Recovery"
[recovery2-cropped]: ./images/track1-recovery2-cropped.jpg "Recovery"
[recovery3-cropped]: ./images/track1-recovery3-cropped.jpg "Recovery"
[recovery4-cropped]: ./images/track2-recovery1-cropped.jpg "Recovery"
[recovery5-cropped]: ./images/track2-recovery2-cropped.jpg "Recovery"
[recovery6-cropped]: ./images/track2-recovery3-cropped.jpg "Recovery"
[video1]: ./video.mp4 "Video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* process.py contains methods for augmenting data and loading images
* preprocess.py contains code for removing extra data from the training set
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Assuming that the training data is in folder data in the current directory the model can be trained using:

```sh
python model.py
```

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed


My model is described on lines 80-94 of model.py and it is inspired by the NVidia self driving car paper. The convolution layers are:

| Layer | Filter Size | Depth |
|-------|-------------|-------|
|   1   |   5x5       |  24   |
|   2   |   5x5       |  36   |
|   3   |   3x3       |  48   |
|   4   |   3x3       |  64   |

Each of the convolution layers have RELU activation to introduce nonlinearity and there is max pooling between each of the layers with a filter size of 2x2.

After that there is are 3 fully connected layers each with 500, 100 and 20 neurons each and RELU activation units between each layer.

To avoid overfitting I added dropout layers between each of these fully connected layers.

#### 2. Attempts to reduce overfitting in the model

I used dropout layers in order to avoid overfitting. The model was also trained with a training set and a validation model was used.

The model was trained for a total of 30 epochs. After that the error in the validation set did not really go down any more.

The validation and training tests are defined in line 71 and training and validation generators are defined in line 74 and 75.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I made the vehicle cross the bridge 3 times to ensure that it could go around the track. I did have some issues initially trying to run it in my computer. It is an old laptop without a video car so my model was overshooting because the was some lag in the images the simulator was sending. My car would start to oscillate left and right faster and faster.

I solved this issue by selecting "fastest" from the simulator graphics options and slownig the speed of the car to 9. I believe that my model could probably do better if tested on a faster machine.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

To collect the data I drove the car around the track using the arrow keys. I believe probably driving with the mouse would have been better. My hypothesis is that when you are not pressing the keys, the steering angle is 0. While driving with the arrow keys there were long streches where I would not press the keys then pressed then for less than a second just making small corrections. I imagine that the binary nature of key pressing doesn't translate well to creating smooth data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get inspiration from the lenet model and to emulate the Nvidia model. My intution is that the images in our track are less complex than what the Nvidia model had so I could make a simpler model.

Initially I tried to use the full size of the Nvidia network, however it was taking incredibly long to train and my training program was crashing. I devided to simplify my network and have only 2 Convolutional layers instead of 4. However, the although the training data had a low mse, the validation set still had a relatively high mse. I think that with a relatively simple architecture it was easy to overfit quickly.

Another reason for the quick overfitting is that I did not have enough data to generalize well. I would go back and gather more driving data then train again.

I went back to 4 layers but added pooling layers to reduce the number of parameters in my network. A pooling layer downsamples the output of the previous layer. I also added dropout layers to prevent overfitting.

In the example writeup report they describe the same problem I was having with my car. My graphics were probably set too high. My car would start driving correctly but would overshoot on the angle. This would start to grow and eventually would drive off the road. I found out that the graphics might have been set too high for my computer. I adjusted for that and also decreased the driving speed to 9. That seemed to work and my car was able to drive a few times around track 1.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-95) consisted of a convolution neural network with the following layers and layer sizes:

Here is a description of my network

| Layer | Type | Description |
| ------|------|-------------|
|  1   | Conv | filter: 5x5 depth: 24  with RELU|
|  2   | MaxPool | filter: 2x2 |
|  3   | Conv | filter: 5x5 depth: 36 with RELU|
|  4   | MaxPool | filter: 2x2 |
|  5   | Conv | filter: 3x3 depth: 48 with RELU|
|  6   | MaxPool | filter: 2x2 |
|  7   | Conv | filter: 3x3 depth: 64 with RELU|
|  8   | MaxPool | filter: 2x2 |
| 9    | Fully Connected| 500 with RELU |
| 10    | Fully Connected| 100 with RELU |
| 11    | Fully Connected| 20 with RELU |
| 12    | Output | 1 output |

#### 3. Creation of the Training Set & Training Process

I first captured good driving behavior by driving in the center of the track several times on track 1 and track 2. Here are two examples.

![alt text][track1]
![alt text][track2]

I also captured a few recoery phases from the edge of the road to get back to the center. Driving in game is not my strength so in addition to that there were a few times where I would overshoot and also have to recover durin ga normal drive. Here are a few examples of that:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]
![alt text][recovery4]
![alt text][recovery5]
![alt text][recovery6]

I augmented the images by flipping vertically and randomly generating a shadow. Here are examples of such a procedure:

![alt text][track1-processed]
![alt text][track2-processed]
![alt text][recovery1-processed]
![alt text][recovery2-processed]
![alt text][recovery3-processed]
![alt text][recovery4-processed]
![alt text][recovery5-processed]
![alt text][recovery6-processed]

I also cropped the images here are the images cropped:

![alt text][track1-cropped]
![alt text][track2-cropped]
![alt text][recovery1-cropped]
![alt text][recovery2-cropped]
![alt text][recovery3-cropped]
![alt text][recovery4-cropped]
![alt text][recovery5-cropped]
![alt text][recovery6-cropped]

Because of the way I was driving using the keyboard, most of the steering angles were zeros. I decided to even out the examples more and removed most of the zero steering angles. In the end I was left with about 8000 images.

The data collection tool has a left and right camera which increase that number by 3 to about 24000. During training I also added random shadows to my examples. This has the benefit of increasing the total number of examples exponentially. I generated the shadows on the fly during training to avoid overfitting. This gave better results than simply generating all the possible images and then training on that set. It also mean I didn't have to store all those images.

I shuffled my data and used 20% of it in the validation set. I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 30 determined by trying epochs 10, 20, 30, 40. The 30th epoch seemed to perform better with the validation data as well perform well with the test track. I used an adam optimizer so that manually training the learning rate wasn't necessary.

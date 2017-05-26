[//]: # (Image References)


[ml]: ./img/machine_learning.png "The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent."
[cd]: ./img/cd.jpg "Center Driving"
[rd]: ./img/rd.jpg "Right side recovery"
[ld]: ./img/ld.jpg "Left side recovery"
[cd_flip]: ./img/cd_flip.jpg "Flipped Image: Center Driving"

#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I modified the PI controller such that the integral part is only updated when the actual vehicle speed is close to the setpoint to avoid overshooting and "winding up".

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I'm using the model architecture described in the Nvidia paper "End to End Learning for Self-Driving Cars".

The model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 55-63) 

My model includes RELU activations after the convulutional layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 53). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 62-72). 

The model was trained and validated on split data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72). The correction angles for using the left and right camera images as well were tuned starting from the value used in the project intro (0.2). I chose the final value of .11 based on the validation results of the model.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I added extra recordings of tight curves and the bridge to get more data on these situations.

###Model Architecture and Training Strategy

####1. Solution Design Approach

As there has been a lot of heavy lifting already done at Nvidia, I decided to start with what I knew from their published paper was a working system. 

In order to gauge how well the model was working, I used 80% of my image and steering angle data as training set and used the remaining data as validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added Dropout layers with a 25% percent drop rate to the higher convolutional layers. This led to a decrease in validation loss showing better generalization than before.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. These were the bridge, where my model appeared to fail completely and the right turn. I introduced flipping the images to combat the latter with good success. For the bridge my solution was to add more data. I had tried to record a number of recovery laps on the bridge, which apperently were more confusing than helping the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 61-75) consisted of a convolution neural network with the following layers and layer sizes:

* 24x5x5 Convolution, 2x2 stride, Relu activation
* 36x5x5 Convolution, 2x2 stride, Relu activation
* 48x5x5 Convolution, 2x2 stride, Relu activation
* 64x3x3 Convolution, Relu activation
* 64x3x3 Convolution, Relu activation
* 100 Fully Connected
* 50 Fully Connected
* 10 Fully Connected
* Output node

That is also the starting configuration taken from the Nvidia paper.

![][ml]

https://xkcd.com/1838/

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][cd]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from veering off to the sides of the track. These images show what a recovery looks like starting from each side of the track.

![alt text][rd]
![alt text][ld]

To augment the data sat, I also flipped images and angles thinking that this would combat the left turn bias of the closed track. For example, here is the center driving image shown above that has then been flipped:

![alt text][cd_flip]

After the collection process, I had 11679 images. I then preprocessed this data by normalising it (model.py line 58) and cropping off the top and bottom which do not show the road (model.py line 60).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I found the number of epochs to be best set at five, as the validation loss did not decrease further after additional epochs or even increased again. I used an adam optimizer so that manually training the learning rate wasn't necessary.

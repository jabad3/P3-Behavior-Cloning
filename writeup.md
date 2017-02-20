#**Behavioral Cloning** 

##Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[centerLaneDriving]: ./examples/centerLaneDriving.jpg "Alt"
[errorControl]: ./examples/errorControl.jpg "Alt"
[noFlip]: ./examples/noFlip.jpg "Alt"
[yesFlip]: ./examples/yesFlip.jpg "Alt"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create the model
* train.py containing the script to train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.json containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for creating the model and the train.py file contains the code that trains the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based of a convolution neural network similar to the one used in second project. The model has five layers (model.py lines 8-35). See the following section for details on its structure.

The model includes dropout layers, max-pooling, ELU layers, and flattened layers to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in order to reduce overfitting (model.py lines 14,20,27). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py file code line 16-17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 37).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and drift control data. Drift control data is similar to recovery data, but focuses on preventing the car from drifting. I also generated extra data for spots where the model was having difficulty (tight turns and detecting concrete vs dirt). Unfortunetly, correcting for these situations introduced some noise into my model, as can be seen when the car approaches corners.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was first to read the [NVIDEA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that deals with this problem. Next I used their similar model to test the results on my dataset. I think this model is appropriate because it correctly captures behavior that allows the car to drive on the track.

There were two spots on the first track that proved to be tricky at first. To combat this, I generated more data for those spots, and made sure to train very carefull for that behavior.

Very notable in my solution is that I only used data from the first track to train the model. I did not augment the color space of the image. However, the model does not seem to be overfitting because when tested on a completly different track (track 2, different color space and different driving pattern) it performs very well (maybe even better than the first track).  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture
The final model architecture (model.py lines 7-37) consisted of a convolution neural network with the following layers and layer sizes:
First Layer: Convolution layer with 32 5x5 filters. This layer ends with an ELU activation.
Second Layer: Convolution layer with 16 3x3 filters. This layer also has an ELU activation, dropout of 40%, and a max pool layer.
Third Layer: Convolution layer with 16 3x3 filters. This layer has an ELU activation and a dropout set to 40% as well.
Fourth Layer: Fully connected layer with 1024 perceptrons. A dropout layer is set to 30%, and an ELU activation is at the end.
Fifth Layer: Fully connected layer with 1024 perceptrons. Another dropout layer set to 30%, and an ELU activation.


####3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][centerLaneDriving]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to  correct its trajectory if it made a mistake. This image shows an example of a slow recovery when the car got too close to the left hand side of the lane:

![alt text][errorControl]


To augment the data sat, I also flipped images and angles. This serves two purposes, it easily creates new data to train on, and second, it helps to reduce bias for the curves.

After the collection process, I had approximatly 73,000 images (about 210,000 total after flipping and augmenting). I then preprocessed this data by cropping the bottom of the image (the car hood was visible) and the top of the image (anything above the horizon is not needed and adds noise). Finally, I reduced the size of the image to 64x64 in order to reduce computation and complexity.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 5 epochs. After this, the error was not decreasing by much. I also used an adam optimizer so that manually training the learning rate was not necessary.

####Credit:
I used [this](https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686#.de3l6yu4h) blog that provided tips and insights into this problem to help steer my project.

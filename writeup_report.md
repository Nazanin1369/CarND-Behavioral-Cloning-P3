#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/raw_samples-histogram.png "Raw data histogram"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/augment_flipped_center_2016_12_01_13_44_57_783.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* myutils.py containing the code for data loading and augmentation
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started with with a simple model with 2 convolutions and a fully-connected layer, just to test that my code is working.   Once everything started to fit in together, I finally adapted the model used by NVIDIA as described in this paper  - https://arxiv.org/abs/1604.07316.

My model consists of a convolution neural network having 3 convolution layers with 5x5 filters and another 2 convolution layers with 3x3 filters. This is followed by 3 fully-connected layers and a final layer for the steering angle predicted value (model.py lines 60-92).

The model includes ELU layers to introduce nonlinearity (code line 66, 68, 70, 73, 75), and the data is normalized in the model using a Keras lambda layer (code line 62). 

As suggested in the lessons, I added a Cropping2D layer that is useful for choosing an area of interest that excludes the sky and/or the hood of the car (code line 63). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers with a rate of 0.3 in order to reduce overfitting (model.py lines 79, 81). 

The model was trained and validated on the Udacity-provided data and data I generated as part of augmentation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer.  The initial learning rate was set to 0.001 (model.py line 23, 88).

####4. Appropriate training data

Training data used is that from Udacity plus additional data created in augmentation stage. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

As mentioned earlier, I adapted the model used by NVIDIA.

With my current setup, this model took time for training. I decided to remove the 1164-neuron Dense layer to reduce parameters and hopefully reduce training time.   This change did not negatively affect the performance of my model so I settled with this final model.   


####2. Final Model Architecture

The final model architecture (model.py lines 60-92) consisted of a convolution neural network with the following layers and layer sizes ...

Layer (type)                     Output Shape          Param #     Connected to                     

====================================================================================================

lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             

____________________________________________________________________________________________________

cropping2d_1 (Cropping2D)        (None, 75, 320, 3)    0           lambda_1[0][0]                   

____________________________________________________________________________________________________

convolution2d_1 (Convolution2D)  (None, 36, 158, 24)   1824        cropping2d_1[0][0]               

____________________________________________________________________________________________________

elu_1 (ELU)                      (None, 36, 158, 24)   0           convolution2d_1[0][0]            

____________________________________________________________________________________________________

convolution2d_2 (Convolution2D)  (None, 16, 77, 36)    21636       elu_1[0][0]                      

____________________________________________________________________________________________________

elu_2 (ELU)                      (None, 16, 77, 36)    0           convolution2d_2[0][0]            

____________________________________________________________________________________________________

convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       elu_2[0][0]                      

____________________________________________________________________________________________________

elu_3 (ELU)                      (None, 6, 37, 48)     0           convolution2d_3[0][0]            

____________________________________________________________________________________________________

convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       elu_3[0][0]                      

____________________________________________________________________________________________________

elu_4 (ELU)                      (None, 4, 35, 64)     0           convolution2d_4[0][0]            

____________________________________________________________________________________________________

convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       elu_4[0][0]                      

____________________________________________________________________________________________________

elu_5 (ELU)                      (None, 2, 33, 64)     0           convolution2d_5[0][0]            

____________________________________________________________________________________________________

flatten_1 (Flatten)              (None, 4224)          0           elu_5[0][0]                      

____________________________________________________________________________________________________

dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]                  

____________________________________________________________________________________________________

dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    

____________________________________________________________________________________________________

dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  

____________________________________________________________________________________________________

dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    

____________________________________________________________________________________________________

dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  

____________________________________________________________________________________________________

dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    

====================================================================================================

Total params: 559,419

Trainable params: 559,419

Non-trainable params: 0


A visualization of the architecture can be found here: https://arxiv.org/pdf/1604.07316.pdf


####3. Creation of the Training Set & Training Process

I ended up just using the Udacity training data as my baseline.   But looking at the distribution of the steering angles, it looks like there is an abundance of zero or near-zero angles.  This means that there are more data driving in straight than curved lanes.  This is not ideal since our model learning could end up having a bias on straight line driving.  We need to filter this data.  Below is the distribution of the raw data.

![alt text][image1]

I was also curious what part of the track generated higher steering angles.   Here are samples of steering angles above 0.9.


The image on the bridge looks like an anomaly as this is a stright line.  This could have been a result of the driver making a quick steering adjustment in this part of the track.  To eliminate this cases, I decided to filter also steering angles greater than 0.9.  Here is the distribution after the filter. 

I tried the model with the data after filtering.  For this first attempt, the car did not complete the track and would veer off the drivable portion of the track.   I attribute this to the reduced number of data available for the model to learn from.  Clearly, I would have to add more by augmenting.

   
The simulator captures images from three cameras mounted on the car: a center, right and left camera.  I used the left and right images with a steering correction of 0.28.

            correction = 0.28
            steering_left = steering_center + correction
            steering_right = steering_center - correction

I utilized the augmentation ideas and code from Vivek Yadav's work.   However, I only performed these augmentation for images with steering angles greater than 0.20. The purpose of this is to add more representation from higher angled steering to improve the distribution of the data. Below are sample images:
Adding ramdom brightness brightness:


Adding ramdom shadow:

Adding horizontal shift


Flipping

After performing these augmentation, we ended up with x samples.  After adding the flipped images, our data distribution looks like these.   It is still not the ideal balance of data I was going for but looks like these can work.

I used a 80-20 split on the data, 20% being used for validation.

Working with images takes a huge chunk of memory to be used especially if the images are loaded all at the same time.  To avoid hitting memory limitations, I used Keras generators to provide just-in-time loading of the image files that are fed to the model during training.   I used a batch size of 32 samples.

I set the number of epochs to 6.  Surprisingly, the best performance came out of epoch number 2.


![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

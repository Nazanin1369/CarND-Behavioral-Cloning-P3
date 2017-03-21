import os
import csv
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

LEARNING_RATE = .0001
EPOCHS = 5
STEERING_CORRECTION = 0.28

MODEL_FILE = 'model.h5'
DRIVING_LOG = './data/driving_log.csv'

def read_driving_log(filename):
    raw_samples = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            raw_samples.append(line)

    return raw_samples[1:]

# remove data closer to zero angle to improve data distribution
def filter_straight_line(samples):
    filtered = []

    for s in samples:
        steering_center = float(s[3])

        if (abs(steering_center) < 0.01):
            continue
        else:
            filtered.append(s)

    print('Num samples after filtering data: ', len(filtered))  

    return filtered          

# Use the left and right camera images 
# Use the center camera steering angle and add a correction factor
def augment_with_lf_camera(samples):
    augmented_sample = []

    for s in samples:
        steering_center = float(s[3])

        augmented_sample.append([s[0], steering_center])

        correction = STEERING_CORRECTION
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        augmented_sample.append([s[1], steering_left])
        augmented_sample.append([s[2], steering_right])

    print('Num samples after adding left/right camera: ', len(augmented_sample)) 

    return augmented_sample

# taken from Vivek Yadav
def change_brightness(image):
    # Randomly select a percent change
    change_pct = np.random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_image

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = batch_sample[1]

                images.append(center_image)
                angles.append(center_angle)

                # add data for higher angle steering
                if (abs(center_angle) > .30 ):
                    curve_img = change_brightness(center_image)
                    images.append(curve_img)
                    angles.append(center_angle)

                    images.append(np.fliplr(curve_img))
                    angles.append(center_angle * -1.0)

                # Add flipped image and steering
                flipped = np.fliplr(center_image)
                images.append(flipped)
                angles.append(center_angle * -1.0)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Model used is based on NVIDIA as described at
# https://arxiv.org/abs/1604.07316
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))

    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1164))
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Using the ADAM optimizer with a custom starting learning rate
    optimizer = Adam(lr=LEARNING_RATE)

    model.compile(loss='mse', optimizer=optimizer)

    return model    


def main():

    t_samples, v_samples = train_test_split(read_driving_log(DRIVING_LOG), test_size=0.2)

    print('Number of train samples     : ', len(t_samples))
    print('Number of validation samples: ', len(v_samples))

    train_samples = augment_with_lf_camera(filter_straight_line(t_samples))
    validation_samples = augment_with_lf_camera(filter_straight_line(v_samples))

 
    print('train_samples[0]', train_samples[0])

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=128)
    validation_generator = generator(validation_samples, batch_size=128)

    model = get_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # http://machinelearningmastery.com/check-point-deep-learning-models-keras/
    # https://keras.io/callbacks/#create-a-callback
    checkpointer = ModelCheckpoint(
        filepath="weights.best.hdf5", verbose=1, save_best_only=True)

    print("Training the model ...")

    history_object = model.fit_generator(train_generator, samples_per_epoch =
        len(train_samples) * 3, validation_data = 
        validation_generator, callbacks=[early_stopping, checkpointer],
        nb_val_samples = len(validation_samples) * 3, 
        nb_epoch=EPOCHS, verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    print("Saving the model file=", MODEL_FILE)

    #Save the model
    model.save(MODEL_FILE)

if __name__ == "__main__":
    main()
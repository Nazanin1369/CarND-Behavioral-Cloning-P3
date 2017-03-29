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

from myutils import *

LEARNING_RATE = .0001
EPOCHS = 1

MODEL_FILE = 'model.h5'

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                name = batch_sample.directory +  batch_sample.image_file

                image = cv2.imread(name)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                angle = batch_sample.steering_angle

                images.append(image)
                angles.append(angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Model used is based on NVIDIA as described at
# https://arxiv.org/abs/1604.07316
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,25),(0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(36,5,5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample=(2,2)))
    model.add(ELU())

    model.add(Convolution2D(64,3,3))
    model.add(ELU())
    model.add(Convolution2D(64,3,3))
    model.add(ELU())

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    # Using the ADAM optimizer with a custom starting learning rate
    optimizer = Adam(lr=LEARNING_RATE)

    model.compile(loss='mse', optimizer=optimizer)

    return model    



def main():

    samples = get_driving_data(read_driving_log(DRIVING_LOG))

    train_samples, validation_samples = train_test_split(augment_and_visualize_data(samples), test_size=0.2)

    print('Number of train samples     : ', len(train_samples))
    print('Number of validation samples: ', len(validation_samples))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = get_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # http://machinelearningmastery.com/check-point-deep-learning-models-keras/
    # https://keras.io/callbacks/#create-a-callback
    checkpointer = ModelCheckpoint(
        filepath=MODEL_FILE + "-epoch-{epoch:02d}", verbose=1, save_best_only=True)

    print("Training the model ...")

    samples_per_epoch = len(train_samples)
    #samples_per_epoch = en(train_samples) * 3
    history_object = model.fit_generator(train_generator, samples_per_epoch =
        samples_per_epoch, validation_data = 
        validation_generator, callbacks=[early_stopping, checkpointer],
        nb_val_samples = len(validation_samples), 
        nb_epoch=EPOCHS, verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    print("Saving the model file=", MODEL_FILE)

    #Save the model
    model.save(MODEL_FILE)

if __name__ == "__main__":
    main()
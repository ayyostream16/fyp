import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    0:"1run",
    1:"2runs",
    2:"3runs",
    3:"4runs",
    4:"5runs",
    5:"6runs",
    6:"none"
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


# load images from the directory
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
        '/Users/balajim/Desktop/wd/rock-paper-scissors/image_data',
        target_size=(150, 150),
        batch_size=15,
        class_mode='categorical')







# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=5)
       
# save the model for later use
model.save("rock-paper-scissors-model.h5")

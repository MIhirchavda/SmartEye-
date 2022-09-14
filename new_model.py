import os
import numpy as np
import cv2


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.optimizers import Adam
from keras.models import load_model

base = 'E:\OpenCv\Face_Recognition1\H5_data set'
train_set = os.path.join(base,'E:\OpenCv\Face_Recognition1\H5_data set\Training_set')
validation_set = os.path.join(base,'E:\OpenCv\Face_Recognition1\H5_data set\Test_set')


example_list = [(os.path.join(train_set, folder)) for folder in os.listdir('E:\OpenCv\Face_Recognition1\H5_data set\Training_set') if os.path.isdir(os.path.join(train_set, folder)) == True]
title_list = [folder for folder in os.listdir(train_set) if os.path.isdir(os.path.join(train_set, folder)) == True]
fig = plt.figure(figsize=(20, 10))
for i, img_path in enumerate(example_list):
    sp = plt.subplot(2, 4, i + 1)
    plt.title(title_list[i])
    img = cv2.imread(os.path.join(example_list[i], os.listdir(img_path)[0]))
    img = cv2.resize(img, (182, 182))
    plt.imshow(img)

plt.show()

padding = 'valid'

img_input = layers.Input(shape=(64, 64, 1))

# START MODEL
conv_1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
maxpool_1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_1)
x = layers.BatchNormalization()(maxpool_1)

# FEAT-EX1
conv_2a = layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', padding=padding, name='conv_2a')(x)
conv_2b = layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
maxpool_2a = layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_2a')(x)
conv_2c = layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_2c')(maxpool_2a)
concat_1 = layers.concatenate(inputs=[conv_2b, conv_2c], axis=3, name='concat2')
maxpool_2b = layers.MaxPooling2D((3, 3), strides=(2, 2), padding=padding, name='maxpool_2b')(concat_1)

# FEAT-EX2
conv_3a = layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
conv_3b = layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
maxpool_3a = layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_3a')(maxpool_2b)
conv_3c = layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_3c')(maxpool_3a)
concat_3 = layers.concatenate(inputs=[conv_3b, conv_3c], axis=3, name='concat3')
maxpool_3b = layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_3b')(concat_3)

# FINAL LAYERS
net = layers.Flatten()(maxpool_3b)
net = layers.Dense(5, 'softmax', name='predictions')(net)

model = Model(img_input, net)

model.summary()
model.compile(optimizer = Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'] )


train_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

train = train_generator.flow_from_directory(train_set,target_size = (64,64),color_mode = 'grayscale',batch_size = 1,class_mode = 'categorical')
validation = validation_generator.flow_from_directory(validation_set,target_size = (64,64),color_mode = 'grayscale',batch_size = 1,class_mode = 'categorical')

output = model.fit(train,steps_per_epoch = 1485,epochs = 25,validation_data = validation,validation_steps = 1080,verbose = 1)
model.save("hew_model.h5")
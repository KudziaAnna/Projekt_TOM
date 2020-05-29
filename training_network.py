# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:58:35 2020

@author: Ania
"""
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

#model from https://github.com/qubvel/segmentation_models

batch_size = 200
target_size = (128,128)
num_dev_samples =2896
num_train_samples = 38782 

#%%Preparing data generators for model training
path_train_X = 'images/train/VOL'
path_dev_X = 'images/dev/VOL'
path_train_Y = 'images/train/SEG'
path_dev_Y = 'images/dev/SEG'

image_train_datagen =  ImageDataGenerator()
mask_train_datagen = ImageDataGenerator()
image_dev_datagen =  ImageDataGenerator()
mask_dev_datagen = ImageDataGenerator()

seed =1
image_train_generator = image_train_datagen.flow_from_directory(
    directory = path_train_X,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)

mask_train_generator = mask_train_datagen.flow_from_directory(
    directory = path_train_Y,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)

image_dev_generator = image_dev_datagen.flow_from_directory(
    directory = path_dev_X,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)

mask_dev_generator = mask_dev_datagen.flow_from_directory(
    directory = path_dev_Y,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_train_generator, mask_train_generator)
dev_generator = zip(image_dev_generator, mask_dev_generator)

#%%Loss function for image segmentation
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef2(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    score = 0.5 * score0 + 0.5 * score1+0.5*score2

    return score


def dice_coef_loss(y_true, y_pred):
    return -dice_coef2(y_true, y_pred)

#%%Defining and training Unet model
model = Unet('mobilenet',encoder_weights='imagenet', classes = 3, input_shape=(None, None, 3))

optimizer = Adam(lr=3e-05,beta_1=0.9, beta_2=0.999, decay=0.00)
model.compile(optimizer= optimizer, loss=dice_coef_loss,
              metrics=['accuracy'])
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch = num_train_samples/batch_size,
                    epochs = 10,
                    validation_data = dev_generator,
                    validation_steps = num_dev_samples/batch_size)

model.save_weights('Unet_mobilenet_weights.h5')
model.save('Unet_mobile_model')
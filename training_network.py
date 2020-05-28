# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:58:35 2020

@author: Ania
"""

from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss
from segmentation_models.metrics import FScore
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.config.experimental.list_physical_devices('GPU')

batch_size = 200
target_size = (128,128)
num_dev_samples =2896
num_train_samples = 38782 

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

#%%
model = Unet('resnet34',encoder_weights='imagenet', classes = 3, input_shape=(128, 128, 3),)
loss = CategoricalCELoss()
metrics = FScore()
model.compile('Adam', loss=loss, metrics=[metrics])
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch = num_train_samples/batch_size,
                    epochs = 10,
                    validation_data = dev_generator,
                    validation_steps = num_dev_samples/batch_size)

model.save_weights('nn_weights.h5')
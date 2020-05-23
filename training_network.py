# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:58:35 2020

@author: Ania
"""

from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss
from segmentation_models.metrics import FScore
from keras.preprocessing.image import ImageDataGenerator

batch_size = 5
target_size = (256,256)
num_dev_samples =5

path_train_X = 'img_cache/vol/train_X'
path_dev_X = 'img_cache/vol/dev_X'
path_train_Y = 'img_cache/seg/train_Y'
path_dev_Y = 'img_cache/seg/dev_Y'

image_train_datagen =  ImageDataGenerator()
mask_train_datagen = ImageDataGenerator()
image_dev_datagen =  ImageDataGenerator()
mask_dev_datagen = ImageDataGenerator()

seed =1
image_train_generator = image_train_datagen.flow_from_directory(
    path_train_X,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)
mask_train_generator = mask_train_datagen.flow_from_directory(
    path_train_Y,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)
image_dev_generator = image_dev_datagen.flow_from_directory(
    path_dev_X,
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)
mask_dev_generator = mask_dev_datagen.flow_from_directory(
    target_size = target_size,
    batch_size = batch_size,
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_train_generator, mask_train_generator)
dev_generator = zip(image_dev_generator, mask_dev_generator)

#%%
model = Unet('resnet34',encoder_weights='imagenet', classes = 3, input_shape=(256, 256, 3),)
loss = CategoricalCELoss()
metrics = FScore()
model.compile('Adam', loss=loss, metrics=[metrics])
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch = 40,
                    epochs = 10,
                    validation_data = dev_generator,
                    validation_steps = num_dev_samples/batch_size)

#model.save_weights('nn_weights.h5')
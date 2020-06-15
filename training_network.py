from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
import numpy as np

BATCH_SIZE = 16
TARGET_SIZE = (256,256)
NUM_DEV_SAMPLES = 8126
NUM_TRAIN_SAMPLES = 74500
EPOCHS = 30
H = 256
W = 256
CHANNEL = 3


path_train_X = 'train/VOL'
path_dev_X = 'dev/VOL'
path_train_Y = 'train/SEG'
path_dev_Y = 'dev/SEG'

#%%Defining dice_loss function and metric dice_coef2
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef2(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    score = score0*0.1+score1*0.3+score2*0.6
    return score


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef2(y_true, y_pred)

#%%Defining  Unet model
def get_unet():
    inputs = Input((H, W, CHANNEL))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

    up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

    conv12 = Conv2D(3, (1, 1), activation=('softmax'), name='conv10')(conv11)
    return inputs, conv12

#%% Importing and normalizig data
image_train_datagen =  ImageDataGenerator(rescale = 1./255)
mask_train_datagen = ImageDataGenerator(rescale = 1/127, dtype = 'int')
image_dev_datagen =  ImageDataGenerator(rescale = 1./255)
mask_dev_datagen = ImageDataGenerator(rescale = 1/127, dtype = 'int')

seed =1

image_train_generator = image_train_datagen.flow_from_directory(
    directory = path_train_X,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode=None,
    seed=seed)
mask_train_generator = mask_train_datagen.flow_from_directory(
    directory = path_train_Y,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode=None,
    seed=seed)
image_dev_generator = image_dev_datagen.flow_from_directory(
    directory = path_dev_X,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode=None,
    seed=seed)
mask_dev_generator = mask_dev_datagen.flow_from_directory(
    directory = path_dev_Y,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_train_generator, mask_train_generator)
dev_generator = zip(image_dev_generator, mask_dev_generator)

#%% Training model

inputs, final = get_unet()
model = Model(inputs=[inputs], outputs=[final])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
optimizer = Adam(lr=3e-04,beta_1=0.9, beta_2=0.999, decay=0.2)
model.compile(optimizer= optimizer, loss=dice_coef_loss,
              metrics=[dice_coef2])
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch = NUM_TRAIN_SAMPLES/BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = dev_generator,
                    validation_steps = NUM_DEV_SAMPLES/BATCH_SIZE,
                    callbacks=[callback],
                    verbose = 1)

model.save('Unet_model')

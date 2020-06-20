# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:42:44 2020

@author: Ania
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import numpy as np
import nibabel as nib
import os, glob

BATCH_SIZE = 16
TARGET_SIZE = (256,256)
NUM_TEST_SAMPLES = [730, 292,  322, 698, 266, 466, 270,196,356,324,538,242,206,192,766,1240,150,186,120,280,178,204]

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f).numpy()
    print('intersection ' + str(intersection))
    union =( K.sum(y_true_f) + K.sum(y_pred_f) ).numpy()
    print('union ' +str(union))
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef2(y_true, y_pred):
    
    y_true_0 = tf.where(y_true ==0, 1,0)
    y_pred_0 = tf.where(y_pred ==0, 1,0)
    y_true_1 = tf.where(y_true ==1, 1,0)
    y_pred_1 = tf.where(y_pred ==1, 1,0)
    y_true_2 = tf.where(y_true ==2, 1,0)
    y_pred_2 = tf.where(y_pred ==2, 1,0)
    
    score0 = dice_coef(y_true_0, y_pred_0)
    score1 = dice_coef(y_true_1, y_pred_1)
    score2 = dice_coef(y_true_2, y_pred_2)
    
    dice_kidney.append(score1)
    dice_tumor.append(score2)
    score = score0*0.1+score1*0.3+score2*0.6
    
    return score


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef2(y_true, y_pred)

#%%
seg_path = 'kits19/data'
vol_path = os.path.join('images/test', 'case*')
vol_files = glob.glob(vol_path)

image_test_datagen =  ImageDataGenerator(rescale = 1./255)
seed =1

model_path = 'Unet_model'
model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef2':dice_coef2})
#%%
dice =[]
dice_kidney =[]
dice_tumor =[]
predictions =[]
seg =[]
tmp =0
for i in vol_files:
        test_generator = image_test_datagen.flow_from_directory(
            directory =os.path.join(i, 'vol'),
            target_size = (256,256),
            batch_size = BATCH_SIZE,
            class_mode=None,
            seed=seed)
        
        prediction = model.predict(test_generator, 
                                   steps = NUM_TEST_SAMPLES[tmp]/BATCH_SIZE,
                                   verbose=1)
    
        predict_class = np.argmax(prediction, axis=3)
  
        
        mask_directory = os.path.join(seg_path,  'case_00'+str(188+tmp),'segmentation.nii.gz')
        mask = nib.load(mask_directory)
        mask = mask.get_data()
        
        #rescale = mask.shape[0]/predict_class.shape[0]
        #predict_class = zoom(predict_class, (0.5,2,2), order =1, mode='nearest')
        #mask = zoom(mask, (2,0.5, 0.5))

        seg.append(mask)
        predictions.append(predict_class)
        
       mask  = tf.convert_to_tensor(mask, dtype=tf.int32)
    
       dice_tmp = dice_coef2(mask, predict_class)
       dice.append(dice_tmp)
       
        tmp+=1
    
 #%%
dice_mean = np.mean(dice)
dice_max = np.max(dice)
dice_min = np.min(dice)
dice_std = np.std(dice)

dice_mean_kidney = np.mean(dice_kidney)
dice_max_kidney = np.max(dice_kidney)
dice_min_kidney = np.min(dice_kidney)
dice_std_kidney = np.std(dice_kidney)

dice_mean_tumor = np.mean(dice_tumor)
dice_max_tumor = np.max(dice_tumor)
dice_min_tumor = np.min(dice_tumor)
dice_std_tumor = np.std(dice_tumor)

print(dice)

print("Dice max: "+str(dice_max))
print("Dice min: "+str(dice_min))
print("Dice mean: "+str(dice_mean))
print("Dice standard deviation: "+str(dice_std))

print("Dice max for kidney: "+str(dice_max_kidney))
print("Dice min for kidney: "+str(dice_min_kidney))
print("Dice mean for kidney: "+str(dice_mean_kidney))
print("Dice standard deviation for kidney: "+str(dice_std_kidney))

print("Dice max for tumor: "+str(dice_max_tumor))
print("Dice min for tumor: "+str(dice_min_tumor))
print("Dice mean for tumor: "+str(dice_mean_tumor))
print("Dice standard deviation for tumor: "+str(dice_std_tumor))


#%%
best_dice = (predictions[12]*127).astype(np.uint8)
best_mask = (seg[12]*127).astype(np.uint8)
worst_dice = (predictions[15]*127).astype(np.uint8)
worst_mask = (seg[15]*127).astype(np.uint8)
average_dice = (predictions[5]*127).astype(np.uint8)
average_mask = (seg[5]*127).astype(np.uint8)

#%%
fig = plt.figure(figsize=(500, 500))

plt.suptitle('Best results')
ax1 = fig.add_subplot(1,2,1, projection='3d' )
ax1.voxels(best_dice )
ax1.set_title('Our results')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.voxels(best_mask)
ax2.set_title('True results')
plt.show()


fig = plt.figure(figsize=(500, 500))
plt.suptitle('Worst results')
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.voxels(worst_dice)
ax1.set_title('Our results')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.voxels(worst_mask)
ax1.set_title('True results')
plt.show()


fig = plt.figure(figsize=(500, 500))
plt.suptitle('Average results')
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.voxels(average_dice)
ax1.set_title('Our results')
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.voxels(average_mask)
ax1.set_title('True results')
plt.show()



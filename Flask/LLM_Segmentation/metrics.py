# metrics.py

import tensorflow as tf
import keras.backend as K

def dice_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + 1) / (union + 1)
    return 1 - dice

def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + 1) / (union + 1)
    return dice
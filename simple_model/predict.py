import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

import utils as ut
#import unet as un
#import models as mod

from tqdm import tqdm
from itertools import chain

from random import seed
from random import randint

 
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
 
#tf.enable_eager_execution()
 
IMG_WIDTH = 320
IMG_HEIGHT = 240
DOFS = 7 # from 7 # 3 + 3 + 3 or 3 + 4
NBDATA = 100

# the directtory train/ contains the directory images/ and masks/
TRAIN_PROJ_PATH = 'outputTest/proj/'
TRAIN_POSE_PATH = 'outputTest/pose/'

seed(1)

X = ut.loadRawData(TRAIN_PROJ_PATH, IMG_WIDTH, NBDATA)
y = ut.loadPoseData(TRAIN_POSE_PATH, DOFS, NBDATA)

X = X/255.
X = np.expand_dims(X, axis=-1)
y = np.asarray(y)


print('Data loaded')
print(X.shape)
print(y.shape)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

    
#Predict on train, val and test
model = load_model('model.h5')

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])

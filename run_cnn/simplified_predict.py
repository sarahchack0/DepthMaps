import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

# import unet as un
# import models as mod

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
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
# from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

# tf.enable_eager_execution()

IMG_WIDTH = 256
IMG_HEIGHT = 192
DOFS = 9 # change from 3
NBDATA = -1
MODE = 0 # if 1 train on all the set, otherwise split the set into training and validation

def loadDictData(pickled_dict_path):
    # for Y, should return np matrix of n camera positions
    # p p p v v v u u u (n times)

    unpickled_dict = {}

    with open(pickled_dict_path, 'rb') as f:
        unpickled_dict = pickle.load(f)

    pos_list = []
    depth_list = []
    for key in unpickled_dict:
        # save all camera positions to list (first entry in the list value in dictionary)
        temp = []
        for sublist in unpickled_dict[key][0]:
            for tuple_val in sublist:
                temp.append(tuple_val)
        pos_list.append(temp)

        depth_list.append(unpickled_dict[key][1])

    # convert list to np array, with shape of (number of maps, 9)
    pos_matrix = np.array(pos_list)

    # each individual depth map is (768. 1024). whole matrix shape should be (number of maps, 768, 1024)
    depth_matrix = np.array(depth_list)

    print(pos_matrix.shape) # with dict of 15 maps, shape is (15, 9)
    print(depth_matrix.shape) # with dict of 15 maps, shape is (15, 768, 1024) # (, 1)

    #print(pos_matrix.view(dtype=np.int16, type=np.matrix))

    return depth_matrix, pos_matrix


# the directtory train/ contains the directory images/ and masks/
# TRAIN_PROJ_PATH = 'outputTest/proj/'
# TRAIN_POSE_PATH = 'outputTest/pose/'

seed(1)
#
# X = ut.loadRawData(TRAIN_PROJ_PATH, IMG_WIDTH, NBDATA)
# y = ut.loadPoseData(TRAIN_POSE_PATH, DOFS, NBDATA)

pickled_dict_filename = "../training_data/pickled_dict.pkl"
X, y = loadDictData(pickled_dict_filename)

# Split train and valid
if MODE == 1:
    X_train = X
    y_train = y
else:
    # splits between train and test
    # keep 25% of data for validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

X = X / 1.0
X = np.expand_dims(X, axis=-1)  # add 1 to the end
y = np.asarray(y)

print('Data augmented')
print(X.shape)
print(y.shape)

# Predict on train, val and test using the saved model
model = load_model('model.h5')

ix = randint(0, X_valid.shape[0])
pred = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(pred)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
pred = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(pred)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
pred = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(pred)
print(y_valid[ix])

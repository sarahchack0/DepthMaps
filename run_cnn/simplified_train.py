from __future__ import print_function, absolute_import, division, unicode_literals

import random
import pickle

import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#import utils as ut

from tqdm import tqdm
from itertools import chain

from random import seed
from random import randint

 
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import elasticdeform

from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import cv2
import re
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
IMG_WIDTH = int(1024/4)
IMG_HEIGHT = int(768/4)

overall_max = -999
overall_min = 999

first = 0

def loadDictData(pickled_dict_path):
    global overall_max, overall_min, first
    # for Y, should return np matrix of n camera positions
    # p p p v v v u u u (n times)

    unpickled_dict = {}

    with open(pickled_dict_path, 'rb') as f:
        unpickled_dict = pickle.load(f)

    pos_list = []
    depth_list = []

    
    for key in unpickled_dict:
        tmp = unpickled_dict[key][1]
        # start with filler values -999, so find max num
        if np.max(tmp) > overall_max:
          overall_max = np.max(tmp)
        
        tmp[tmp == -999] = 999

        # changed to filler values 999, so find min num
        if np.min(tmp) < overall_min:
          overall_min = np.min(tmp)
    print("overall min, ", overall_min)
    print("overall max, ", overall_max)

    for key in unpickled_dict:
        # save all camera positions to list (first entry in the list value in dictionary)
        temp = []
        for sublist in unpickled_dict[key][0]:
            for tuple_val in sublist:
                temp.append(tuple_val)
        pos_list.append(temp)
        arr = unpickled_dict[key][1]
      
        #arr[arr == -999] = 999
        normed = NormalizeData(arr)
        if first == 0:
            plt.imshow(normed)
            plt.show()
        #print((arr>0).sum())
        img = cv2.resize(normed, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
        if first == 0:
            plt.imshow(img)
            plt.show()
        #print("before norm", img)

        #normed = NormalizeData(img)

        #print("min", np.min(normed))
        #print("after norm", normed)

        # replace any value over 1 with 2
        if first == 0:
            print(img)
        img[img < 0] = 2.0

        if first == 0:
            print(img)
            plt.imshow(img)
            plt.show()

        #print("after replacing filler values", normed)
        

        depth_list.append(img)
        first += 1

    # convert list to np array, with shape of (number of maps, 9)
    pos_matrix = np.array(pos_list)

    # each individual depth map is (768. 1024). whole matrix shape should be (number of maps, 768, 1024)
    plt.imshow(depth_list[0])
    #plt.text(0, 0, str(unpickled_dict[x][0]))
    plt.show()

    # each individual depth map is (768. 1024). whole matrix shape should be (number of maps, 768, 1024)
    depth_matrix = np.array(depth_list)

    print(pos_matrix.shape) # with dict of 15 maps, shape is (15, 9)
    print(depth_matrix.shape) # with dict of 15 maps, shape is (15, 768, 1024) # (, 1)

    #print(pos_matrix.view(dtype=np.int16, type=np.matrix))

    return depth_matrix, pos_matrix

def NormalizeData(data):
    #return ((data - overall_min) / (overall_max - overall_min))
    return data / -1000


DOFS = 9 # change from 3
NBDATA = -1
MODE = 0 # if 1 train on all the set

random.seed(1)

pickled_dict_filename = "../training_data/dict1_small.pkl"
X, y = loadDictData(pickled_dict_filename)

print('Data loaded')
print(X.shape) # X should equal Y
print(y.shape)

X = X/1.0
X = np.expand_dims(X, axis=-1) # add 1 to end
y = np.asarray(y)
    
print('Data augmented')
print(X.shape)
print(y.shape)    



# Split train and valid
if MODE == 1:
    X_train = X
    y_train = y
else:
    # splits between train and test
    # keep 25% of data for validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)


fig = plt.figure(figsize=(40,8))    

ix = randint(0, X_train.shape[0])   
ax1 = fig.add_subplot(1,1,1)
ax1.imshow(np.squeeze(X_train[ix]))
ax1.set_title(str(y_train[ix]))
#plt.show()




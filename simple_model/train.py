 
from __future__ import print_function, absolute_import, division, unicode_literals
import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import utils as ut

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
#from keras import backend as K


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True



#tf.enable_eager_execution()

# smaller images make training easier
#
IMG_WIDTH = 1024
IMG_HEIGHT = 768
DOFS = 9 # change from 3
NBDATA = -1
MODE = 0 # if 1 train on all the set

caseID = '20180814'

# the directtory train/ contains the directory images/ and masks/
TRAIN_PROJ_PATH = './Data/'+str(caseID)+'/proj/' # 3D maps, image
TRAIN_POSE_PATH = './Data/'+str(caseID)+'/pose/' # pose, file with matrix that you store

seed(1)

# X = ut.loadRawData(TRAIN_PROJ_PATH, IMG_HEIGHT, IMG_WIDTH, NBDATA)
# y = ut.loadPoseData(TRAIN_POSE_PATH, DOFS, NBDATA)

pickled_dict_filename = "../training_data/pickled_dict.pkl"
X, y = ut.loadDictData(pickled_dict_filename)

print('Data loaded')
print(X.shape) # X should equal Y
print(y.shape)


# probability mapping augmentation
for i in range(0, len(X)): # adding some noise
    X = np.append(X, np.expand_dims(ut.syntetizeProba(X[i]), axis=0), axis=0)
    y = np.append(y, np.expand_dims(y[i], axis=0), axis=0)


# deformation augmentation
for i in range(0, len(X)): # deform images with elastic deform, keep normal and deformed
    # apply deformation with a random 3 x 3 grid
    X = np.append(X, np.expand_dims(elasticdeform.deform_random_grid(X[i], sigma=25, points=3), axis=0), axis=0)
    y = np.append(y, np.expand_dims(y[i], axis=0), axis=0)


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





inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))

c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
#c1 = Dropout(0.1) (c1)
#c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
#c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
#c2 = Dropout(0.2) (c2)
#c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
#c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
#c3 = Dropout(0.2) (c3)
#c3 = BatchNormalization()(c3)
c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
#c3 = BatchNormalization()(c3)

c4 = Flatten()(c3)
c5 = Dense(128, activation='relu')(c4)
outputs = Dense(DOFS, activation='linear')(c5)


model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

if MODE == 1:
    results = model.fit(X_train, y_train, batch_size=8, epochs=30)
    model.save("model_def_proba.h5")
else:    
    callbacks = [
        EarlyStopping(patience=15, verbose=1, min_delta=0.0001),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    ]

    results = model.fit(X_train, y_train, batch_size=8, epochs=40, callbacks=callbacks, validation_data=(X_valid, y_valid))

    f = open(str(caseID)+'_errors.txt', 'w')

    for i in range(0, len(X_valid)):
        p = model.predict(np.expand_dims(X_valid[i], axis=0), verbose=1)
        err = ut.compute_ADD_t(y_valid[i],p, './real/'+str(caseID)+'/'+str(caseID)+'_Vessels_ablation_000.obj')
        f.write(str(err)+'\n')
        #print(err)

    f.close() 

    # Evaluate on validation set (this must be equals to the best log_loss)
    print(model.evaluate(X_valid, y_valid, verbose=1))


    # Get actual number of epochs model was trained for
    N = len(results.history['loss'])

    #Plot the model evaluation history
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(40,8))

    fig.add_subplot(1,1,1)
    plt.title("Training Loss")
    plt.plot(np.arange(0, N), results.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",label="best model")

    plt.show()

    np.set_printoptions(suppress=True)



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




import os
import sys
import numpy as np
import skimage as sk
import random
import cv2

from tqdm import tqdm
from itertools import chain
 
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import pickle

from scipy.spatial.transform import Rotation as R
import re

# length is number of images you want to look at
    # - 1 means all images
def loadRawData(disk_path, hsize, wsize, lenght = -1):    
    sys.stdout.flush()
    nbImg = len(os.listdir(disk_path))
    if (lenght == -1 or lenght > nbImg):
        lenght = nbImg
    
    x = np.zeros((lenght, hsize, wsize), dtype=np.int8)

    for n, img in enumerate(tqdm(os.listdir(disk_path))):
        if n == lenght:
            break            
        path = os.path.join(disk_path,img)
        if path.endswith(".png"):
            img = imread(path, as_gray=True)
            #img = resize(img, (hsize, wsize), mode='constant', preserve_range=True).astype(np.int8)
            img = cv2.resize(img, (wsize, hsize), interpolation = cv2.INTER_AREA)
        
            x[n] = img
        
    return x

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


#loadDictData("../training_data/pickled_dict.pkl")

def loadPoseData(disk_path, dofs, lenght = -1):    
    sys.stdout.flush()
    nbItems = len(os.listdir(disk_path))
    if (lenght == -1 or lenght > nbItems):
        lenght = nbItems
    
    x = np.zeros((lenght,dofs), dtype=np.float)
    

    for n, item in enumerate(tqdm(os.listdir(disk_path))):
        if n == lenght:
            break
        path = os.path.join(disk_path,item)
        if path.endswith(".txt"):
            file = open(path, 'r')
            t = np.zeros(3, dtype=np.float)
            q = np.zeros(4, dtype=np.float)
            for line in file.readlines():
                line = line.strip()
                pose = [float(n) for n in line.split(' ')]
                q[0] = pose[0]
                q[1] = pose[1]
                q[2] = pose[2]
                q[3] = pose[3]
                t[0] = pose[4]
                t[1] = pose[5]
                t[2] = pose[6]
                
            #x[n] = pose
            x[n] = t
            #x[n] = q
        
    return x    


def syntetizeProba(binary):
    rdn = np.random.random(binary.shape[:2])
    rdn[np.where(binary < 1)] = 0
    
    return rdn

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice



def compute_ADD(y, p, objFilename):
    
    q1 = [] 
    q1.append(y[0])
    q1.append(y[1])
    q1.append(y[2])
    q1.append(y[3])
    t1 = []
    t1.append(y[4])
    t1.append(y[5])
    t1.append(y[6])

    q2 = [] 
    q2.append(p[0][0])
    q2.append(p[0][1])
    q2.append(p[0][2])
    q2.append(p[0][3])
    t2 = []
    t2.append(p[0][4])
    t2.append(p[0][5])
    t2.append(p[0][6])
    
    reComp = re.compile("(?<=^)(v |vn |vt |f )(.*)(?=$)", re.MULTILINE)
    with open(objFilename) as f:
        data = [txt.group() for txt in reComp.finditer(f.read())]

    v_arr = []
    for line in data:
        tokens = line.split(' ')
        if tokens[0] == 'v':
            v_arr.append([float(c) for c in tokens[1:]])    


    r1 = (R.from_quat(q1)).as_matrix()
    r2 = (R.from_quat(q2)).as_matrix()

    dist = 0
    for pt in v_arr:
        pt1 = np.dot(r1, pt) + t1
        pt2 = np.dot(r2, pt) + t2

        dist += np.linalg.norm(pt1 - pt2)

    err = dist/len(v_arr)   
    
    
    return err


def compute_ADD_t(y, p, objFilename):
    
    t1 = []
    t1.append(y[0])
    t1.append(y[1])
    t1.append(y[2])

    t2 = []
    t2.append(p[0][0])
    t2.append(p[0][1])
    t2.append(p[0][2])
    
    reComp = re.compile("(?<=^)(v |vn |vt |f )(.*)(?=$)", re.MULTILINE)
    with open(objFilename) as f:
        data = [txt.group() for txt in reComp.finditer(f.read())]

    v_arr = []
    for line in data:
        tokens = line.split(' ')
        if tokens[0] == 'v':
            v_arr.append([float(c) for c in tokens[1:]])    

    r1 = np.identity(3)
    r2 = np.identity(3)

    dist = 0
    for pt in v_arr:
        pt1 = np.dot(r1, pt) + t1
        pt2 = np.dot(r2, pt) + t2


        dist += np.linalg.norm(pt1 - pt2)

    err = dist/len(v_arr)   
    
    
    return err
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:59:10 2019

@author: iaaraya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from classify_models import Metrics, FF
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import sys

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='1' # gpu='0' o gpu='1'
###################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True#Utiliza la memoria que necesita de manera dinamica, puede ser o no en bloque.
config.gpu_options.per_process_gpu_memory_fraction = 0.5#20%de la ram,
session = tf.Session(config=config)

mode = sys.argv[1]
imbalance = sys.argv[2]
X_train = np.loadtxt('X_train_full')
y_train = np.loadtxt('y_train_full')
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)


scalerX = StandardScaler().fit(X_train)

X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

X_train_original = X_train.copy()
y_train_hot_original = y_train_hot.copy()

np.random.seed(42)
if imbalance == 'undersampling':
    unstable_X = X_train[y_train==1]
    unstable_y = y_train_hot[y_train==1]
    stable_X = X_train[y_train==0]
    stable_y = y_train_hot[y_train==0]
    
    stable_index = np.random.choice(len(stable_X),len(unstable_X)*0.1, replace=False)
    stable_X = stable_X[stable_index]
    stable_y = stable_y[stable_index]
    
    print(len(stable_X),len(unstable_X))
    X_train = np.concatenate((stable_X,unstable_X))
    y_train_hot = np.concatenate((stable_y,unstable_y))
    

if mode == 'FF':
    layers = sys.argv[3]
    layers = [int(layer) for layer in layers.split(',')]
    lr = float(sys.argv[4])
    epochs = int(sys.argv[5])
    
    model = FF(layers,X_train.shape[1], lr)
    model.fit(X_train,y_train_hot,validation_data=(X_train_original, y_train_hot_original),epochs=epochs, batch_size=32, \
                verbose=True, callbacks=[Metrics()], shuffle=True)






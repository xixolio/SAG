# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:59:10 2019

@author: iaaraya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from classify_models import Metrics, FF, LSTM_model
from utils import upsampling, downsampling, data_to_sequences
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
import sys
import pandas as pd
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu='0' o gpu='1'
###################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True#Utiliza la memoria que necesita de manera dinamica, puede ser o no en bloque.
config.gpu_options.per_process_gpu_memory_fraction = 0.8#20%de la ram,
session = tf.Session(config=config)


mode = sys.argv[1]
imbalance = sys.argv[2]
length = int(sys.argv[3])

X_train = np.loadtxt('X_train_full')
y_train = np.loadtxt('y_train_full')
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)


scalerX = StandardScaler().fit(X_train)

X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

#X_train,_,_ = standarization(X_train,500)
#X_test,_,_ = standarization(X_test,500)

#y_train = y_train[1:]
#y_test = y_test[1:]
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

if length > 1:
    X_train = data_to_sequences(X_train, length)
    y_train = y_train[length-1:]
    y_train_hot = y_train_hot[length-1:]
    
    X_test = data_to_sequences(X_test, length)
    y_test = y_test[length-1:]
    y_test_hot = y_test_hot[length-1:]
    
    if mode == 'FF':
        X_train = X_train.reshape(X_train.shape[0],-1)
        X_test = X_test.reshape(X_test.shape[0],-1)
    
    print(X_train.shape)
    print(y_train.shape)
        
    
X_train_original = X_train.copy()
y_train_hot_original = y_train_hot.copy()

np.random.seed(42)

if imbalance == 'upsampling':
    X_train, y_train_hot = upsampling(X_train, y_train, y_train_hot)
elif imbalance == 'downsampling':
    X_train, y_train_hot = downsampling(X_train, y_train, y_train_hot)

if mode == 'FF':
    
    argv_pos = 4
    layers = sys.argv[argv_pos]
    layers = [int(layer) for layer in layers.split(',')]
    lr = float(sys.argv[argv_pos+1])
    epochs = int(sys.argv[argv_pos+2])
    
    model = FF(layers,X_train.shape[1], lr)
    model.fit(X_train,y_train_hot,validation_data=(X_test, y_test_hot),epochs=epochs, batch_size=32, \
                verbose=True, callbacks=[Metrics()], shuffle=True)

elif mode == 'LSTM':
    argv_pos = 4
    layers = sys.argv[argv_pos]
    layers = [int(layer) for layer in layers.split(',')]
    lr = float(sys.argv[argv_pos+1])
    epochs = int(sys.argv[argv_pos+2])
    
    model = LSTM_model(layers,X_train.shape[-1], length, lr)
    model.fit(X_test,y_test_hot,validation_data=(X_test, y_test_hot),epochs=epochs, batch_size=32, \
                verbose=True, callbacks=[Metrics()], shuffle=True)




# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:00:18 2019

@author: iaaraya
"""

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):

        val_predict = np.argmax(self.model.predict(self.validation_data[0]),axis=1)
        val_target = np.argmax(self.validation_data[1],axis=1)
        
        val_f1 = f1_score(val_target, val_predict, average="weighted")
        val_recall = recall_score(val_target, val_predict, average=None)
        val_precision = precision_score(val_target, val_predict, average=None)
        
        print(val_f1)
        print(val_recall)
        print(val_precision)
        return
    
   
def FF(layers,features,lr, dropout=None):
    inputs = Input(shape=(features,))
    dense = inputs
    #dense = Dropout(0.1)(dense)
    for hidden in layers:
        if dropout:
            dense = Dropout(dropout)(dense)
        dense = Dense(hidden,activation='relu', kernel_initializer='glorot_uniform')(dense)
    
    if dropout:
            dense = Dropout(dropout)(dense)
    outputs = Dense(2,activation='softmax')(dense)
    model = Model(inputs = inputs, outputs = outputs)
    
    ad = Adam(lr = lr)    
    model.compile(loss = 'categorical_crossentropy', optimizer = ad)
    return model

def LSTM_model(layers,features, length,lr):
    inputs = Input(shape=(length,features))
    
    lstm = inputs
    for blocks in layers:
        lstm = LSTM(blocks)(lstm)
        
    outputs = Dense(2,activation='softmax')(lstm)
    model = Model(inputs = inputs, outputs = outputs)
    
    ad = Adam(lr = lr)    
    model.compile(loss = 'categorical_crossentropy', optimizer = ad)
    return model
        
        
    
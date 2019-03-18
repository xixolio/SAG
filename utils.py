# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:08:13 2019

@author: iaaraya
"""
import numpy as np
import pandas as pd
""" DATA PREPROCESSING """

def data_to_sequences(X, length=1):
    data = []
    for i in range(X.shape[0]-length+1):
        data.append(X[i:i+length,:].reshape(1,length,-1))    
    return np.concatenate(data,axis=0)

def standarization(data, past_horizon):
    #print(len(data))
    data = pd.DataFrame(data.copy())
    mean_values = pd.DataFrame()
    std_values = pd.DataFrame()
    for col in range(data.shape[1]):
        mean_values[col] = data[col].rolling(past_horizon,min_periods=1).mean()
        std_values[col] = data[col].rolling(past_horizon,min_periods=1).std()
        std_values[std_values[col] == 0] = 1
        data[col] = (data[col] - mean_values[col])/std_values[col]
    
    data = data.dropna()
    mean_values = mean_values[1:]
    std_values = std_values[1:]

    return data.values, mean_values.values, std_values.values

""" DATA BALANCE """

def downsampling(X_train,y_train,y_train_hot):
    unstable_X = X_train[y_train==1]
    unstable_y = y_train_hot[y_train==1]
    stable_X = X_train[y_train==0]
    stable_y = y_train_hot[y_train==0]
    
    stable_index = np.random.choice(len(stable_X),int(len(unstable_X)), replace=False)
    stable_X = stable_X[stable_index]
    stable_y = stable_y[stable_index]
    
    X_train = np.concatenate((stable_X,unstable_X))
    y_train_hot = np.concatenate((stable_y,unstable_y))
    return X_train, y_train_hot

def upsampling(X_train,y_train,y_train_hot):
    
    unstable_X = X_train[y_train==1]
    unstable_y = y_train_hot[y_train==1]
    stable_X = X_train[y_train==0]
    stable_y = y_train_hot[y_train==0]
    
    proportion = int(len(stable_X)/len(unstable_X))

    unstable_X = np.repeat(unstable_X,proportion,axis=0)
    unstable_y = np.repeat(unstable_y,proportion,axis=0)
    
    X_train = np.concatenate((stable_X,unstable_X))
    y_train_hot = np.concatenate((stable_y,unstable_y))
    return X_train, y_train_hot


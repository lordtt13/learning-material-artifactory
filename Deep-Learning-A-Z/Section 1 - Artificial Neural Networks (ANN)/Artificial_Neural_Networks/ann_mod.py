# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 00:55:45 2019

@author: tanma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

def convert(x):
    return tf.cast(x,tf.int32)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = convert(sc.fit_transform(X_train))
X_test = convert(sc.transform(X_test))
y_train = convert(y_train)
y_test = convert(y_test)

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class CustomConstraint(Constraint):
    def __call__(self,w):
        w += K.cast(w,tf.int32)
        return w

def my_init(shape, dtype = None):
    return K.cast(K.random_normal(shape, dtype=dtype),tf.int32)

inp_ = Input(shape = (11,))
x = Dense(units = 6, kernel_initializer = my_init, bias_initializer = my_init, activation = 'relu', kernel_constraint = CustomConstraint(), bias_constraint = CustomConstraint())(inp_)
op = Dense(units = 1, kernel_initializer = my_init, bias_initializer = my_init, activation = 'relu', kernel_constraint = CustomConstraint(), bias_constraint = CustomConstraint())(x)

model = Model(inp_,op)
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 1, epochs = 10)
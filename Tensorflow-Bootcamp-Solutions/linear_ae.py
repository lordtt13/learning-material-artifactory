# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:21:05 2019

@author: tanma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('anonymized_data.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('Label',axis=1))

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

num_inputs = 3
num_hidden = 2  
num_outputs = num_inputs 
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize( loss)

init = tf.global_variables_initializer()

num_steps = 1000

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train,feed_dict={X: scaled_data})
        
with tf.Session() as sess:
    sess.run(init)
        
    output_2d = hidden.eval(feed_dict={X: scaled_data})


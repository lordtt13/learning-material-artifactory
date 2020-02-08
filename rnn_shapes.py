# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:28:01 2020

@author: Tanmay Thakur
"""

import tensorflow as tf
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, SimpleRNN, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


# Make some data
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)

# Make an RNN
M = 5 # number of hidden units
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i, x)

Yhat = model.predict(X)
print(Yhat)

model.summary()

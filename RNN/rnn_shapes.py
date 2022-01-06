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

Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

h_last = np.zeros(M) # initial hidden state
x = X[0] # the one and only sample
Yhats = [] # where we store the outputs

for t in range(T):
  h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
  y = h.dot(Wo) + bo # we only care about this value on the last iteration
  Yhats.append(y)
  
  # important: assign h to h_last
  h_last = h

# print the final output
print(Yhats[-1])
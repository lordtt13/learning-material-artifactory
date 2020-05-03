#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 06:23:48 2020

@author: tanmay
"""

import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


# Create Sine Wave

x = torch.linspace(0,799, steps = 800)
y = torch.sin(x*2*3.1416/40)

plt.figure(figsize = (12,4))
plt.xlim(-10,801)
plt.grid(True)
plt.plot(y.numpy())

test_size = 40

# Split the dataset

train_set = y[:-test_size]
test_set = y[-test_size:]

# Create time series

def input_data(seq, ws):  # ws is the window size
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

window_size = 40

# Create the training dataset of sequence/label tuples:
train_data = input_data(train_set,window_size)

len(train_data)

# Display the first (seq/label) tuple in train_data
train_data[0]

torch.set_printoptions(sci_mode = False) # to improve the appearance of tensors
train_data[0]
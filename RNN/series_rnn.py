#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 06:36:08 2020

@author: tanmay
"""

import time
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Data/TimeSeriesData/Alcohol_Sales.csv',index_col = 0,parse_dates = True)
len(df)

df.dropna(inplace = True)
len(df)

df.head()

# Plot Dataset

plt.figure(figsize = (12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis = 'x',tight = True)
plt.plot(df['S4248SM144NCEN'])
plt.show()

# Extract values from the source .csv file
y = df['S4248SM144NCEN'].values.astype(float)

# Define a test size
test_size = 12

# Create train and test sets
train_set = y[:-test_size]
test_set = y[-test_size:]

scaler = MinMaxScaler(feature_range = (-1, 1))

train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

# Convert train_norm from an array to a tensor
train_norm = torch.FloatTensor(train_norm).view(-1)

# Define a window size
window_size = 12

# Define function to create seq/label tuples
def input_data(seq,ws):  # ws is the window size
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

# Apply the input_data function to train_norm
train_data = input_data(train_norm,window_size)
len(train_data)  # this should equal 325-12-12
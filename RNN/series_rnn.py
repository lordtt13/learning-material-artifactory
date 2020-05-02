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
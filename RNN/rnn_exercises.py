#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 06:40:58 2020

@author: tanmay
"""

import time
import torch

import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


df = pd.read_csv('Data/TimeSeriesData/Energy_Production.csv', index_col = 0, parse_dates = True)
df.dropna(inplace = True)
print(len(df))
df.head()

# Plot data series

plt.figure(figsize = (12,4))
plt.title('Industrial Production Index for Electricity and Gas Utilities')
plt.ylabel('Index 2012 = 100, Not Seasonally Adjusted')
plt.grid(True)
plt.autoscale(axis = 'x',tight = True)
plt.plot(df['IPG2211A2N'])
plt.show()


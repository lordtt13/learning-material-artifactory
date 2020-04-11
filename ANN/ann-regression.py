#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:30:45 2020

@author: tanmay
"""

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Visualize the dataset
df = pd.read_csv('../Data/NYCTaxiFares.csv') # Dataset from Kaggle
df.head()

df['fare_amount'].describe()

# Calculate Distance from Latitude and Longitude
def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d


# Create distance coln
df['dist_km'] = haversine_distance(df,'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df.head()

# Create DateTime coln
df['EDTdate'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour']<12,'am','pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")
df.head()

df['EDTdate'].min(), df['EDTdate'].max()

# Separate categorical from continuous columns
df.columns

cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount'] # labels

# Categorify
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
 
# Check data
df.dtypes

df['Hour'].head()

df['AMorPM'].head()

# Categories
df['AMorPM'].cat.categories, df['AMorPM'].head().cat.codes

df['Weekday'].cat.categories, df['Weekday'].head().cat.codes

# combine the three categorical columns into one input array
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], 1)

cats[:5]

# Convert numpy array to torch tensors
cats = torch.tensor(cats, dtype = torch.int64) 

cats[:5]

# Convert continuous variables to a tensor
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)
conts[:5]

conts.type()

# Convert labels to a tensor
y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1)

y[:5]

# Check shape
cats.shape, conts.shape, y.shape
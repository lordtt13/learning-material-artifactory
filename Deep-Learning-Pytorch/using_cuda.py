#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:02:06 2020

@author: tanmay
"""

import time
import torch

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


torch.cuda.is_available()

torch.cuda.current_device()

torch.cuda.get_device_name(0)

# Returns the current GPU memory usage by 
# tensors in bytes for a given device
torch.cuda.memory_allocated()

# Returns the current GPU memory managed by the
# caching allocator in bytes for a given device
torch.cuda.memory_cached()

# Using CUDA instead of CPU

a = torch.FloatTensor([1.,2.])

a.device

# GPU
a = torch.FloatTensor([1., 2.]).cuda()

a.device

torch.cuda.memory_allocated()

# Send model to gpu

class Model(nn.Module):
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
model = Model()

# From the discussions here: discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda
next(model.parameters()).is_cuda

gpumodel = model.cuda()

next(gpumodel.parameters()).is_cuda

# Load dataset

df = pd.read_csv('../Data/iris.csv')
X = df.drop('target',axis = 1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

# Convert Tensors to .cuda() tensors

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()

trainloader = DataLoader(X_train, batch_size = 60, shuffle = True)
testloader = DataLoader(X_test, batch_size = 60, shuffle = False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 100
losses = []
start = time.time()
for i in range(epochs):
    i+=1
    y_pred = gpumodel.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f'TOTAL TRAINING TIME: {time.time()-start}')
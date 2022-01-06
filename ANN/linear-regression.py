#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 05:37:32 2020

@author: tanmay
"""
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


torch.manual_seed(42)

X = torch.linspace(1,50,50).reshape(-1,1)

# Equivalent to
# X = torch.unsqueeze(torch.linspace(1,50,50), dim=1)

e = torch.randint(-8, 9,(50, 1), dtype = torch.float)
print(e.sum())

y = 2*X + 1 + e
print(y.shape)

plt.scatter(X.numpy(), y.numpy())
plt.ylabel('y')
plt.xlabel('x')
plt.plot()

# Simple Linear Model
model = nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    

# Model Definition
model = Model(1, 1)
print(model)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())
    
# Naive prediction
x = torch.tensor([2.0])
print(model.forward(x))

# Plot initial Model
x1 = np.array([X.min(),X.max()])
print(x1)

w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Initial weight: {w1:.8f}, Initial bias: {b1:.8f}')

y1 = x1*w1 + b1
print(y1) 

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.title('Initial Model')
plt.ylabel('y')
plt.xlabel('x')
plt.plot()

# Set Loss
criterion = nn.MSELoss()

# Set Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 50
losses = []

for i in tqdm(range(epochs)):
    i+=1
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.4f}  weight: {model.linear.weight.item():10.4f}  \
bias: {model.linear.bias.item():10.4f}') 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  
# Plot Losses
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.plot()

# Plot Results
w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Current weight: {w1:.8f}, Current bias: {b1:.8f}')
print()

y1 = x1*w1 + b1
print(x1)
print(y1)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.title('Current Model')
plt.ylabel('y')
plt.xlabel('x')
plt.plot()
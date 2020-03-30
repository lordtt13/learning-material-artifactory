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
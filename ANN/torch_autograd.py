#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 02:13:43 2020

@author: tanmay
"""
import torch


# Single Step
# Requires grad turns on computational tracking
x = torch.tensor(2.0, requires_grad=True)

y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1

print(y)

y.backward()

print(x.grad)

# Multi Step Backprpagation
x = torch.tensor([[1.,2,3],[3,2,1]], requires_grad=True)
print(x)

# First Layer
y = 3*x + 2
print(y)

# Second Layer
z = 2*y**2
print(z)

# Third Layer
out = z.mean()
print(out)

# Output Layer
out.backward()
# Gradient of out wrt x
print(x.grad)
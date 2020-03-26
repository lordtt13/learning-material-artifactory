#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 04:05:18 2020

@author: tanmay
"""

import torch
import numpy as np

torch.__version__


# Numpy Array
arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))

# x here shares same memory as arr, so when you change arr x will change too
# torch.from_numpy and torch.as_tensor provide these functionalities
x = torch.from_numpy(arr)
# Equivalent to x = torch.as_tensor(arr)

print(x)
print(x.dtype)

print(type(x))
print(x.type()) # More Specific

arr2 = np.arange(0., 12,).reshape(4, 3)
print(arr2)

x2 = torch.as_tensor(arr2)
print(x2)
print(x2.type())

# Casting (here) against sharing
# Using torch.tensor()
arr = np.arange(0,5)
t = torch.tensor(arr)
print(t)
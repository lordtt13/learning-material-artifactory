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


"""
Class constructors
torch.Tensor()
torch.FloatTensor()
torch.LongTensor(), etc.

There's a subtle difference between using the factory function torch.tensor(data) and the class constructor torch.Tensor(data).
The factory function determines the dtype from the incoming data, or from a passed-in dtype argument.
The class constructor torch.Tensor()is simply an alias for torch.FloatTensor(data). 
"""

data = np.array([1,2,3])

a = torch.Tensor(data)  # Equivalent to cc = torch.FloatTensor(data)
print(a, a.type())

b = torch.tensor(data)
print(b, b.type())

c = torch.tensor(data, dtype=torch.long)
print(c, c.type())


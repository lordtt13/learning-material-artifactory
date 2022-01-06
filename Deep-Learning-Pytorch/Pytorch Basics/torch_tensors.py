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


"""
Creating tensors from scratch
Uninitialized tensors with .empty()
torch.empty() returns an uninitialized tensor. 
Essentially a block of memory is allocated according to the size of the tensor, and any values already sitting in the block are returned. 
This is similar to the behavior of numpy.empty().
"""

x = torch.empty(4, 3)
print(x)

x = torch.zeros(4, 3, dtype=torch.int64)
print(x)

x = torch.ones(4, 3, dtype = torch.int32)
print(x)

x = torch.arange(0,18,2).reshape(3,3)
print(x)

x = torch.linspace(0,18,12).reshape(3,4)
print(x)


"""
Changing the dtype of existing tensors
Don't be tempted to use x = torch.tensor(x, dtype=torch.type) as it will raise an error about improper use of tensor cloning.
Instead, use the tensor .type() method.
"""

print('Old:', x.type())

x = x.type(torch.int64)

print('New:', x.type())


"""
Random number tensors
torch.rand(size) returns random samples from a uniform distribution over [0, 1)
torch.randn(size) returns samples from the "standard normal" distribution [σ = 1]
Unlike rand which is uniform, values closer to zero are more likely to appear.
torch.randint(low,high,size) returns random integers from low (inclusive) to high (exclusive)
"""

x = torch.rand(4, 3)
print(x)

x = torch.randn(4, 3)
print(x)

x = torch.randint(0, 5, (4, 3))
print(x)


"""
Random number tensors that follow the input size
torch.rand_like(input)
torch.randn_like(input)
torch.randint_like(input,low,high)
these return random number tensors with the same size as input
"""

x = torch.zeros(2,5)
print(x, x.shape)

x2 = torch.randn_like(x)
print(x2, x2.shape)

"""
The same syntax can be used with
torch.zeros_like(input)
torch.ones_like(input)
"""

x3 = torch.ones_like(x2)
print(x3, x3.shape)

# Setting Random Seed
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 02:03:16 2020

@author: tanmay
"""

# 1. Perform standard imports
import torch
import numpy as np

# 2. Set the random seed for NumPy and PyTorch both to "42"
np.random.seed(42)
torch.manual_seed(42)


# 3. Create a NumPy array called "arr" that contains 6 random integers between 0 (inclusive) and 5 (exclusive)
arr = np.random.randint(0,5,6)
print(arr)

# 4. Create a tensor "x" from the array above
x = torch.from_numpy(arr)
print(x)

# 5. Change the dtype of x from 'int32' to 'int64'
x = x.type(torch.int64)
# x = x.type(torch.LongTensor)
print(x.type())

# 6. Reshape x into a 3x2 tensor
x = x.view(3,2)
# x = x.reshape(3,2)
# x.resize_(3,2)
print(x)

# 7. Return the right-hand column of tensor x
print(x[:,1:])
# print(x[:,1])

# 8. Without changing x, return a tensor of square values of x
print(x*x)
# print(x**2)
# print(x.mul(x))
# print(x.pow(2))
# print(torch.mul(x,x))

# 9. Create a tensor "y" with the same number of elements as x, that can be matrix-multiplied with x
y = torch.randint(0,5,(2,3))
print(y)

# 10. Find the matrix product of x and y
print(x.mm(y))
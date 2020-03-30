#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 01:42:48 2020

@author: tanmay
"""
import torch
import numpy as np


x = torch.arange(6).reshape(3,2)
print(x)

# Grabbing the right hand column values
x[:,1]

# Grabbing the right hand column as a (3,1) slice
x[:,1:]


"""
Reshape tensors with .view()
view() and reshape() do essentially the same thing by returning a reshaped tensor without changing the original tensor in place.
"""

x = torch.arange(10)
print(x)

x.view(2,5)

x.view(5,2)

# Views reflect the most current data
z = x.view(2,5)
x[0]=234
print(z)

# Views can infer shape 
x.view(2,-1)

x.view(-1,5)

"""
Adopt another tensor's shape with .view_as()
view_as(input) only works with tensors that have the same number of elements.
"""

x.view_as(z)


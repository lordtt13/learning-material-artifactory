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

# Tensor Arithmetic

a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype=torch.float)
print(a + b)

print(torch.add(a, b))

result = torch.empty(3)
torch.add(a, b, out=result)  # equivalent to result=torch.add(a,b)
print(result)

a.add_(b)  # equivalent to a=torch.add(a,b)
print(a)

# Dot Product

print(a.mul(b)) 
print(a.dot(b))

# Matrix Multiplication

a = torch.tensor([[0,2,4],[1,3,5]], dtype=torch.float)
b = torch.tensor([[6,7],[8,9],[10,11]], dtype=torch.float)

print('a: ',a.size())
print('b: ',b.size())
print('a x b: ',torch.mm(a,b).size())

# Same things

print(torch.mm(a,b))
print(a.mm(b))
print(a @ b)

"""
Matrix multiplication with broadcasting
Matrix multiplication that involves broadcasting can be computed using torch.matmul(a,b) or a.matmul(b) or a @ b
"""

t1 = torch.randn(2, 3, 4)
t2 = torch.randn(4, 5)

print(torch.matmul(t1, t2).size())

# However, the same operation raises a RuntimeError with torch.mm()
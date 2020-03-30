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


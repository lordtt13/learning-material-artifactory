#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:38:10 2020

@author: tanmay
"""

import time
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms


# Transforms

transform = transforms.ToTensor()

# Dataloaders

train_data = datasets.FashionMNIST(root = '../Data', train = True, download = True, transform = transform)
test_data = datasets.FashionMNIST(root = '../Data', train = False, download = True, transform = transform)

class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = False)

# Examine a batch of images

for images,labels in train_loader: 
    break

for images,labels in train_loader: 
    break

print('Label: ', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow = 10)
plt.figure(figsize = (12,4))

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

# Downsampling

##  If a 28x28 image is passed through a Convolutional layer using a 5x5 filter, a step size of 1, and no padding, what is the resulting matrix size?

conv = nn.Conv2d(1, 1, 5, 1)
for x,labels in train_loader:
    print('Orig size:', x.shape)
    break
x = conv(x)
print('Down size:', x.shape)

## If the sample from question 3 is then passed through a 2x2 MaxPooling layer, what is the resulting matrix size?

x = F.max_pool2d(x, 2, 2)
print('Down size:', x.shape)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:26:28 2020

@author: tanmay
"""

import os
import torch
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


warnings.filterwarnings("ignore")

with Image.open('../Data/CATS_DOGS/test/CAT/10107.jpg') as im:
    plt.imshow(im)
    
# Create list of image filenames
    
path = '..\\Data\\CATS_DOGS\\'
img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '\\' + img)
        
print('Images: ',len(img_names))

# Create a DataFrame of image sizes (width x height)

img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
        
print(f'Images:  {len(img_sizes)}')
print(f'Rejects: {len(rejected)}')

df = pd.DataFrame(img_sizes)

# Run summary statistics on image widths
df[0].describe()

# Run summary statistics on image heights
df[1].describe()

# Image Transformations

dog = Image.open('..\\Data\\CATS_DOGS\\train\\DOG\\14.jpg')
print(dog.size)
plt.imshow(dog)

r, g, b = dog.getpixel((0, 0))
print(r,g,b)

"""
transforms.ToTensor()
Converts a PIL Image or numpy.ndarray (HxWxC) in the range [0, 255] to a torch.FloatTensor of shape (CxHxW) in the range [0.0, 1.0]
"""

transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

"""
transforms.Resize(size)
If size is a sequence like (h, w), the output size will be matched to this. If size is an integer, the smaller edge of the image will be matched to this number.
i.e, if height > width, then the image will be rescaled to (size * height / width, size)
"""

transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

small_dog = Image.open('../Data/CATS_DOGS/train/DOG/11.jpg')
print(small_dog.size)
plt.imshow(small_dog)

im = transform(small_dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:26:28 2020

@author: tanmay
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


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

"""
transforms.CenterCrop(size)
If size is an integer instead of sequence like (h, w), a square crop of (size, size) is made.
"""

transform = transforms.Compose([
    transforms.CenterCrop(224), 
    transforms.ToTensor()
])
im = transform(dog) # this crops the original image
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

"""
Other affine transformations
An affine transformation is one that preserves points and straight lines. Examples include rotation, reflection, and scaling. For instance, we can double the effective size of our training set simply by flipping the images.

transforms.RandomHorizontalFlip(p = 0.5)
Horizontally flip the given PIL image randomly with a given probability.
"""

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

"""
transforms.RandomRotation(degrees)
If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
Run the cell below several times to see a sample of rotations.
"""

transform = transforms.Compose([
    transforms.RandomRotation(30),  # rotate randomly between +/- 30 degrees
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

# Pipeline the whole thing

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1),  # normally we'd set p=0.5
    transforms.RandomRotation(30),
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

"""
Normalization
Once the image has been loaded into a tensor, we can perform normalization on it. This serves to make convergence happen quicker during training. The values are somewhat arbitrary - you can use a mean of 0.5 and a standard deviation of 0.5 to convert a range of [0,1] to [-1,1], for example.
However, research has shown that mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] work well in practice.

transforms.Normalize(mean, std)
Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input tensor
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

# Inverse Transformation

inv_normalize = transforms.Normalize(
    mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std = [1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)
plt.figure(figsize = (12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))

plt.figure(figsize = (12,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
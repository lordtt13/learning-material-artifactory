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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:46:14 2020

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
from sklearn.metrics import confusion_matrix 


transform = transforms.ToTensor()

train_data = datasets.MNIST(root = '../Data', train = True, download = True, transform = transform)
train_data

test_data = datasets.MNIST(root = '../Data', train = False, download = True, transform = transform)
test_data

# Examine a training Record
train_data[0]

image, label = train_data[0]
print('Shape:', image.shape, '\nLabel:', label)

# View the image
plt.imshow(train_data[0][0].reshape((28,28)), cmap = "gray")

# Load data using DataLoader
train_loader = DataLoader(train_data, batch_size = 100, shuffle = True)

test_loader = DataLoader(test_data, batch_size = 500, shuffle = False)

# View a batch of images

np.set_printoptions(formatter = dict(int = lambda x: f'{x:4}')) # to widen the printed array

# Grab the first batch of images
for images,labels in train_loader: 
    break

# Print the first 12 labels
print('Labels: ', labels[:12].numpy())

# Print the first 12 images
im = make_grid(images[:12], nrow = 12)  # the default nrow is 8
plt.figure(figsize = (10,4))
# We need to transpose the images from CWH to WHC
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

# Define Model (MLP for CNN)
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz = 784, out_sz = 10, layers = [120,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)
    
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim = 1)
    
model = MultilayerPerceptron()
model

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Flatten the training data for mlp
for images, labels in train_loader:
    print('Batch shape:', images.size())
    break
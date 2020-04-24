#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:59:56 2020

@author: tanmay
"""


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
test_data = datasets.MNIST(root = '../Data', train = False, download = True, transform = transform)

# Create Dataloaders

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = False)
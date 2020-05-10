#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:04:24 2020

@author: tanmay
"""

import time
import torch

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn


with open('Data/shakespeare.txt', 'r', encoding = 'utf8') as f:
    text = f.read()
    
text[:1000]

len(text)

all_characters = set(text)

# Encode Text

decoder = dict(enumerate(all_characters))

encoder = {char: ind for ind,char in decoder.items()}

encoded_text = np.array([encoder[char] for char in text])

encoded_text[:500]



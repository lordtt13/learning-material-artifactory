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


def one_hot_encoder(encoded_text, num_uni_chars):
    '''
    encoded_text : batch of encoded text
    
    num_uni_chars = number of unique characters (len(set(text)))
    '''
    
    # Create a placeholder for zeros.
    one_hot = np.zeros((encoded_text.size, num_uni_chars))
    
    # Convert data type for later use with pytorch (errors if we dont!)
    one_hot = one_hot.astype(np.float32)

    # Using fancy indexing fill in the 1s at the correct index locations
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    

    # Reshape it so it matches the batch sahe
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))
    
    return one_hot

one_hot_encoder(np.array([1,2,0]),3)
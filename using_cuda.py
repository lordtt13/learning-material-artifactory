#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:02:06 2020

@author: tanmay
"""

import torch


torch.cuda.is_available()

torch.cuda.current_device()

torch.cuda.get_device_name(0)

# Returns the current GPU memory usage by 
# tensors in bytes for a given device
torch.cuda.memory_allocated()

# Returns the current GPU memory managed by the
# caching allocator in bytes for a given device
torch.cuda.memory_cached()

# Using CUDA instead of CPU

a = torch.FloatTensor([1.,2.])

a.device

# GPU
a = torch.FloatTensor([1., 2.]).cuda()

a.device

torch.cuda.memory_allocated()


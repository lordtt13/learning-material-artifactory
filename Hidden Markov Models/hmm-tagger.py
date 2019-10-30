# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:47:48 2019

@author: tanma
"""

import matplotlib.pyplot as plt
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)
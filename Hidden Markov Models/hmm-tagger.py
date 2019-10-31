# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:47:48 2019

@author: tanma
"""

import matplotlib.pyplot as plt
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict, namedtuple
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution


data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

assert len(data) == len(data.training_set) + len(data.testing_set), \
       "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"

assert data.N == data.training_set.N + data.testing_set.N, \
       "The number of training + test samples should sum to the total number of samples"

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    d = {}
    for tag, word in zip(sequences_A, sequences_B):
        if tag not in d.keys(): d[tag] = {} 
        if word not in d[tag].keys(): d[tag][word] = 0
        d[tag][word] +=1
    return d

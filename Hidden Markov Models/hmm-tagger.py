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
# Calculate C(t_i, w_i)
# Unwrap the data stream, split it to two lists through zip, reverse the order, split again
emission_counts = pair_counts(*list(zip(*data.training_set.stream()))[::-1])

assert len(emission_counts) == 12, \
       "Uh oh. There should be 12 tags in your dictionary."
assert max(emission_counts["NOUN"], key=emission_counts["NOUN"].get) == 'time', \
       "Hmmm...'time' is expected to be the most common NOUN."

# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word

FakeState = namedtuple("FakeState", "name")

class MFCTagger:
    missing = FakeState(name="<MISSING>")
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))

# Unwrap the data stream, split it to two lists through zip, {word: {tag : count}}
word_counts = pair_counts(*zip(*data.training_set.stream()))

# Using same pattern as the emission test for time
mfc_table = {w: max(word_counts[w], key=word_counts[w].get) for w in word_counts.keys()}

mfc_model = MFCTagger(mfc_table) # Create a Most Frequent Class tagger instance

assert len(mfc_table) == len(data.training_set.vocab), ""
assert all(k in data.training_set.vocab for k in mfc_table.keys()), ""
assert sum(int(k not in mfc_table) for k in data.testing_set.vocab) == 5521, ""


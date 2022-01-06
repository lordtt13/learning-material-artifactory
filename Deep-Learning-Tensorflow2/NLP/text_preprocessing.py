# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:27:05 2020

@author: Tanmay Thakur
"""

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = [
    "I like eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions."
]

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Tokenized sequences
print(sequences)

# Word to tokens 
tokenizer.word_index

# Padded Sequences
data = pad_sequences(sequences)
print(data)

# Setting max sequence length on padding
MAX_SEQUENCE_LENGTH = 5
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)

# Post padding
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print(data)

# Too much padding
data = pad_sequences(sequences, maxlen=6)
print(data)

# Pre truncation
data = pad_sequences(sequences, maxlen=4)
print(data)

# Post truncation
data = pad_sequences(sequences, maxlen=4, truncating='post')
print(data)
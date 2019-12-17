# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:38:42 2019

@author: tanma
"""


import helper

codes = helper.load_data('cipher.txt')
plaintext = helper.load_data('plaintext.txt')

from keras.preprocessing.text import Tokenizer


def tokenize(x):
    """
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """

    x_tk = Tokenizer(char_level=True)
    x_tk.fit_on_texts(x)

    return x_tk.texts_to_sequences(x), x_tk

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
    
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad(x, length=None):
    """
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))
    
def preprocess(x, y):
    """
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_code_sentences, preproc_plaintext_sentences, code_tokenizer, plaintext_tokenizer =\
    preprocess(codes, plaintext)

print('Data Preprocessed')

from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):
    """
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param code_vocab_size: Number of unique code characters in the dataset
    :param plaintext_vocab_size: Number of unique plaintext characters in the dataset
    :return: Keras model built, but not trained
    """
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(plaintext_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_code_sentences, preproc_plaintext_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))

simple_rnn_model = simple_model(
    tmp_x.shape,
    preproc_plaintext_sentences.shape[1],
    len(code_tokenizer.word_index)+1,
    len(plaintext_tokenizer.word_index)+1)

simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, batch_size=32, epochs=5, validation_split=0.2)

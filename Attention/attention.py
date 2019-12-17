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
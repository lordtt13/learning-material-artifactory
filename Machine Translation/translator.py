# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:39:12 2019

@author: Tanmay Thakur
"""

import collections

import helper
import numpy as np
import project_tests as tests

import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector
from keras.layers import concatenate, add, Bidirectional, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy as categorical_crossentropy

from keras.callbacks import TensorBoard
from time import time


# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """   
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        # Find the length of the longest sequence/sentence
        length = max([len(seq) for seq in x])
    
    return pad_sequences(sequences=x, maxlen=length, padding='post')

tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)

def preprocess(x, y):
    """
    Preprocess x and y
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

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=True)

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """    
    input_seq = Input(shape=input_shape[1:])
    rnn = GRU(units=english_vocab_size, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(units=french_vocab_size))(rnn) 
                             
    model = Model(input_seq, Activation('softmax')(logits))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model

tests.test_simple_model(simple_model)

# Pad and Reshape the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size+1,
    french_vocab_size+1)

simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    # Hyperparameters
    embedding_size = 128
    rnn_cells = 200
    dropout = 0.0
    learning_rate = 1e-3
    
    # Sequential Model
    #from keras.models import Sequential
    #model = Sequential()
    #model.add(Embedding(english_vocab_size, embedding_size, input_length=input_shape[1:][0]))
    #model.add(GRU(rnn_cells, dropout=dropout, return_sequences=True))
    #model.add(Dense(french_vocab_size, activation='softmax'))
    #print(model.summary())
    
    # model's Functional equivalent
    input_seq = Input(shape=input_shape[1:])
     
    embedded_seq = Embedding(input_dim = english_vocab_size, 
                             output_dim = embedding_size,
                             input_length=input_shape[1:][0])(input_seq)
    
    rnn = GRU(units=rnn_cells, dropout=dropout, return_sequences=True)(embedded_seq)
    logits = TimeDistributed(Dense(units=french_vocab_size))(rnn) 
    model = Model(input_seq, Activation('softmax')(logits))
    print(model.summary())

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model    
    
tests.test_embed_model(embed_model)

# Pad the input to work with the Embedding layer
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

# Train the neural network 
embed_rnn_model = embed_model(input_shape = tmp_x.shape,
                              output_sequence_length = max_french_sequence_length,
                              english_vocab_size = english_vocab_size+1,
                              french_vocab_size = french_vocab_size+1)


embed_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Hyperparameters
    dropout = 0.0
    learning_rate = 1e-3
    
    # Choose Sequential or Functional API implementation ('seq' or 'func')
    impl='seq'   
    if impl=='func':
        # Sequential Model 
        print("Using Sequential model (Note: this version makes the unitary test to fail: Disable tests to use it)")
        model = Sequential()
        model.add(Bidirectional(GRU(english_vocab_size, dropout=dropout, return_sequences=True)))
        model.add(Dense(french_vocab_size, activation='softmax'))
        
    else:
        # model's Functional equivalent
        # Note : we could have also used "Bidirectional(GRU(...))" instead of buidling the Bidirectional RNNS manually
        print("Using Functional API")
        input_seq = Input(shape=input_shape[1:])
        right_rnn = GRU(units=english_vocab_size, return_sequences=True, go_backwards=False)(input_seq)
        left_rnn = GRU(units=english_vocab_size, return_sequences=True, go_backwards=True)(input_seq)

        # Choose how to merge the 2 rnn layers : add or concatenate
        #logits = TimeDistributed(Dense(units=french_vocab_size))(add([right_rnn, left_rnn])) 
        logits = TimeDistributed(Dense(units=french_vocab_size))(concatenate([right_rnn, left_rnn])) 
        
        model = Model(input_seq, Activation('softmax')(logits))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
  
    return model
   
tests.test_bd_model(bd_model)

# Pad and Reshape the input to work with a RNN without an Embedding layer
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network 
bd_rnn_model = bd_model(input_shape = tmp_x.shape,
                           output_sequence_length = max_french_sequence_length,
                           english_vocab_size = english_vocab_size+1,
                           french_vocab_size = french_vocab_size+1)

print(bd_rnn_model.summary())

bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """    
    # Hyperparameters
    embedding_size = 128
    rnn_cells = 200
    dropout = 0.0
    learning_rate = 1e-3
        
    # Input
    encoder_input_seq = Input(shape=input_shape[1:], name="enc_input")
 
    # Encoder (Return the internal states of the RNN -> 1 hidden state for GRU cells, 2 hidden states for LSTM cells))
    encoder_output, state_t = GRU(units=rnn_cells, 
                                  dropout=dropout,
                                  return_sequences=False,
                                  return_state=True,
                                  name="enc_rnn")(encoder_input_seq)
          #or for LSTM cells: encoder_output, state_h, state_c = LSTM(...)
        
    # Decoder Input   
    decoder_input_seq = RepeatVector(output_sequence_length)(encoder_output)

    # Decoder RNN (Take the encoder returned states as initial states)
    decoder_out = GRU(units=rnn_cells,
                      dropout=dropout,
                      return_sequences=True,
                      return_state=False)(decoder_input_seq, initial_state=state_t)
                                         #or for LSTM cells: (decoder_input_seq, initial_state=[state_h, state_c])
    
    # Decoder output 
    logits = TimeDistributed(Dense(units=french_vocab_size))(decoder_out) 
    
    # Model
    model = Model(encoder_input_seq, Activation('softmax')(logits))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
     
    return model    
    
# Unitary tests
tests.test_encdec_model(encdec_model)

# Pad and Reshape the input to work with the Embedding layer
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network 
encdec_rnn_model = encdec_model(input_shape = tmp_x.shape,
                                output_sequence_length = max_french_sequence_length,
                                english_vocab_size = english_vocab_size+1,
                                french_vocab_size = french_vocab_size+1)
    
print(encdec_rnn_model.summary())

encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2, callbacks=[tensorboard]) 

# Print prediction(s)
print(logits_to_text(encdec_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """    
    # Hyperparameters
    embedding_size = 128
    rnn_cells = 300
    dropout = 0.2
    learning_rate = 1e-3
    
    from keras.layers import LSTM, concatenate
    
    # Input and embedding
    encoder_input_seq = Input(shape=input_shape[1:]) 
    embedded_input_seq = Embedding(input_dim = english_vocab_size,
                                   output_dim = embedding_size,
                                   input_length=input_shape[1:][0])(encoder_input_seq)
    
    ## Note Bidirectional LSTM is used on the encoder side (see https://arxiv.org/pdf/1609.08144.pdf )
    ## Alternate version : Encoder RNN Bidirectional layer (Using the Keras Bidirectional layer wrappers)
    #encoder_output, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(units=rnn_cells,
    #                                                                                                          dropout=dropout,
    #                                                                                                          return_sequences=False,
    #                                                                                                          return_state=True))(embedded_input_seq)
    
    # Encoder Forward RNN layer
    encoder_forward_output, forward_state_h, forward_state_c = LSTM(units=rnn_cells,
                                                                    dropout=dropout,
                                                                    return_sequences=False,
                                                                    return_state=True,
                                                                    go_backwards=False)(embedded_input_seq)
    
    # Encoder backward RNN layer
    encoder_backward_output, backward_state_h, backward_state_c = LSTM(units=rnn_cells,
                                                                       dropout=dropout,
                                                                       return_sequences=False,
                                                                       return_state=True,
                                                                       go_backwards=True)(embedded_input_seq)
    
    # Encoder output and states : Merge the LSTM Forward and Backward ouputs (using 'concatenate' method) 
    state_h = concatenate([forward_state_h, backward_state_h]) 
    state_c = concatenate([forward_state_c, backward_state_c])    
    encoder_output = concatenate([encoder_forward_output, encoder_backward_output])        
         
    # Decoder Input   
    decoder_input_seq = RepeatVector(output_sequence_length)(encoder_output)
    
    # Decoder RNN layer
    # Note : we need twice more LSTM cells as we have concatenated backward and forward LSTM layers in encoder
    decoder_output = LSTM(units=rnn_cells*2,
                                  dropout=dropout,
                                  return_sequences=True,
                                  return_state=False,
                                  go_backwards=False)(decoder_input_seq, initial_state=[state_h, state_c])
    
    # Decoder output
    logits = TimeDistributed(Dense(units=french_vocab_size))(decoder_output) 
    
    # Model
    model = Model(encoder_input_seq, Activation('softmax')(logits))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
    
    return model    
       
tests.test_model_final(model_final)
print('Final Model Loaded\n')

# Train and Print prediction(s)

# Pad the input to work with the Embedding layer
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

# Train the neural network 
final_rnn_model = model_final(input_shape = tmp_x.shape,
                              output_sequence_length = max_french_sequence_length,
                              english_vocab_size = english_vocab_size+1,
                              french_vocab_size = french_vocab_size+1)

print(final_rnn_model.summary(line_length=125))

final_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(final_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    model = model_final(input_shape = x.shape,
                        output_sequence_length = y.shape[1],
                        english_vocab_size = len(x_tk.word_index)+1,
                        french_vocab_size = len(y_tk.word_index)+1)

    model.fit(x, y, batch_size=1024, epochs=20, validation_split=0.2)
    
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))

final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)
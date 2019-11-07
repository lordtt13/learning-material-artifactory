# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:39:07 2019

@author: tanma
"""

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

vocabulary_size = 5000
max_words = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
num_epochs = 3

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]  
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train2, y_train2,
          validation_data=(X_valid, y_valid),
          batch_size=batch_size, epochs=num_epochs)

model.save("rnn_model.h5")

scores = model.evaluate(X_test, y_test, verbose=0)  
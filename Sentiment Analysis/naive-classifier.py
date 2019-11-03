# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:59:34 2019

@author: tanma
"""

import os
import glob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import *
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib # joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays


nltk.download("stopwords")   

def read_imdb_data(data_dir='data'):
    """Read IMDb movie reviews from given directory.
    
    Directory structure expected:
    - data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/
    
    """

    # Data, labels to be returned in nested dicts matching the dir. structure
    data = {}
    labels = {}

    # Assume 2 sub-directories: train, test
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        # Assume 2 sub-directories for sentiment (label): pos, neg
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            # Fetch list of files for this sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            # Read reviews data and assign labels
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)
            
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
    
    # Return data, labels as nested dicts
    return data, labels

data, labels = read_imdb_data()

def prepare_imdb_data(data):
    """Prepare training and test sets from IMDb movie reviews."""
    
    # TODO: Combine positive and negative reviews and labels
    text = data['train']['pos'] + data['train']['neg']
    label = [1]*len(data['train']['pos']) + [0]*len(data['train']['neg'])
    df_train = pd.DataFrame({'text': text, 'label': label}, columns={'text','label'})
    
    text = data['test']['pos'] + data['test']['neg']
    label = [1]*len(data['test']['pos']) + [0]*len(data['test']['pos'])
    df_test = pd.DataFrame({'text': text, 'label': label}, columns={'text','label'})
    
    # TODO: Shuffle reviews and corresponding labels within training and test sets
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    data_train = df_train['text']
    data_test = df_test['text']
    labels_train = df_train['label']
    labels_test = df_test['label']
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


data_train, data_test, labels_train, labels_test = prepare_imdb_data(data)

    """Convert a raw review string into a sequence of words."""
    
    #       Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem
    soup = BeautifulSoup(review.lower())
    text = soup.get_text()
    text = re.sub(r"[^a-zA-Z0-9]"," ", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]
    # Return final list of words
    return words

cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test

# Preprocess data
words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, labels_test)


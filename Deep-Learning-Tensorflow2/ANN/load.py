# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:52:05 2020

@author: Tanmay Thakur
"""
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from google.colab import files, drive


# Using wget:
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.database

!head arrhythmia.database

df = pd.read_csv('arrhythmia.data', header=None)

# since the data has many columns, take just the first few and name them (as per the documentation)
data = df[[0,1,2,3,4,5]]
data.columns = ['age', 'sex', 'height', 'weight', 'QRS duration', 'P-R interval']

plt.rcParams['figure.figsize'] = [15, 15] # make the plot bigger so the subplots don't overlap
data.hist(); # use a semicolon to supress return value

scatter_matrix(data);

# Using tf.keras:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

tf.keras.utils.get_file('auto-mpg.data', url)

df = pd.read_csv('/root/.keras/datasets/auto-mpg.data', header=None, delim_whitespace=True)
df.head()

# Upload to colab:
uploaded = files.upload()
df = pd.read_csv('daily-minimum-temperatures-in-me.csv', error_bad_lines=False)
df.head()

# From google drive:
drive.mount('/content/gdrive')

!ls '/content/gdrive/My Drive/'
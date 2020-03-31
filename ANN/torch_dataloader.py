#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 06:05:01 2020

@author: tanmay
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv('../Data/iris.csv')
df.head()

df.shape

df.describe()

# Plot the Data
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels = labels, loc = 3, bbox_to_anchor = (1.0,0.85))
plt.show()

# Dataset Setup
train_X, test_X, train_y, test_y = train_test_split(df.drop('target',axis=1).values,
                                                    df['target'].values, test_size = 0.2,
                                                    random_state = 42)

X_train = torch.FloatTensor(train_X)
X_test = torch.FloatTensor(test_X)
y_train = torch.LongTensor(train_y).reshape(-1, 1)
y_test = torch.LongTensor(test_y).reshape(-1, 1)

print(f'Training size: {len(y_train)}')
labels, counts = y_train.unique(return_counts=True)
print(f'Labels: {labels}\nCounts: {counts}')

X_train.size()

y_train.size()

data = df.drop('target',axis=1).values
labels = df['target'].values

# Set up a Torch Dataset object
iris = TensorDataset(torch.FloatTensor(data),torch.LongTensor(labels))

len(iris), type(iris)

# Once we have a dataset we can wrap it with a DataLoader. This gives us a powerful sampler that provides single- or multi-process iterators over the dataset.
iris_loader = DataLoader(iris, batch_size = 105, shuffle = True)

for i_batch, sample_batched in enumerate(iris_loader):
    print(i_batch, sample_batched)
    break

# Subscript into the generator, first index is for batch_number, second index is for training data or labels, then count unique items
list(iris_loader)[0][1].bincount()

# Check next item in generator
next(iter(iris_loader))
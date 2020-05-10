#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:04:24 2020

@author: tanmay
"""

import time
import torch

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn


with open('Data/shakespeare.txt', 'r', encoding = 'utf8') as f:
    text = f.read()
    
text[:1000]

len(text)

all_characters = set(text)

# Encode Text

decoder = dict(enumerate(all_characters))

encoder = {char: ind for ind,char in decoder.items()}

encoded_text = np.array([encoder[char] for char in text])

encoded_text[:500]


def one_hot_encoder(encoded_text, num_uni_chars):
    '''
    encoded_text : batch of encoded text
    
    num_uni_chars = number of unique characters (len(set(text)))
    '''
    
    # Create a placeholder for zeros.
    one_hot = np.zeros((encoded_text.size, num_uni_chars))
    
    # Convert data type for later use with pytorch (errors if we dont!)
    one_hot = one_hot.astype(np.float32)

    # Using fancy indexing fill in the 1s at the correct index locations
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    

    # Reshape it so it matches the batch sahe
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))
    
    return one_hot

one_hot_encoder(np.array([1,2,0]),3)


def generate_batches(encoded_text, samp_per_batch = 10, seq_len = 50):
    
    '''
    Generate (using yield) batches for training.
    
    X: Encoded Text of length seq_len
    Y: Encoded Text shifted by one
    
    Example:
    
    X:
    
    [[1 2 3]]
    
    Y:
    
    [[ 2 3 4]]
    
    encoded_text : Complete Encoded Text to make batches from
    batch_size : Number of samples per batch
    seq_len : Length of character sequence
       
    '''
    
    # Total number of characters per batch
    # Example: If samp_per_batch is 2 and seq_len is 50, then 100
    # characters come out per batch.
    char_per_batch = samp_per_batch * seq_len
    
    
    # Number of batches available to make
    # Use int() to roun to nearest integer
    num_batches_avail = int(len(encoded_text)/char_per_batch)
    
    # Cut off end of encoded_text that
    # won't fit evenly into a batch
    encoded_text = encoded_text[:num_batches_avail * char_per_batch]
    
    
    # Reshape text into rows the size of a batch
    encoded_text = encoded_text.reshape((samp_per_batch, -1))
    

    # Go through each row in array.
    for n in range(0, encoded_text.shape[1], seq_len):
        
        # Grab feature characters
        x = encoded_text[:, n:n+seq_len]
        
        # y is the target shifted over by 1
        y = np.zeros_like(x)
       
        #
        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1]  = encoded_text[:, n+seq_len]
            
        # FOR POTENTIAL INDEXING ERROR AT THE END    
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
            
        yield x, y
        
sample_text = encoded_text[:20]

sample_text

batch_generator = generate_batches(sample_text, samp_per_batch = 2, seq_len = 5)

# Grab first batch
x, y = next(batch_generator)

print(x, y)

# GPU Check
torch.cuda.is_available()

# Model Definition

class CharModel(nn.Module):
    
    def __init__(self, all_chars, num_hidden = 256, num_layers = 4, drop_prob = 0.5, use_gpu = False):
        
        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu
        
        #CHARACTER SET, ENCODER, and DECODER
        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: ind for ind,char in decoder.items()}
        
        
        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout = drop_prob, batch_first = True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))
      
    
    def forward(self, x, hidden):
                  
        lstm_output, hidden = self.lstm(x, hidden)
        
        
        drop_output = self.dropout(lstm_output)
        
        drop_output = drop_output.contiguous().view(-1, self.num_hidden)
        
        
        final_out = self.fc_linear(drop_output)
        
        
        return final_out, hidden
    
    
    def hidden_state(self, batch_size):
        '''
        Used as separate method to account for both GPU and CPU users.
        '''
        
        if self.use_gpu:
            
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda(),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        
        return hidden
    
# Instantiate Model

model = CharModel(
    all_chars = all_characters,
    num_hidden = 512,
    num_layers = 3,
    drop_prob = 0.5,
    use_gpu = True,
)

total_param  = []
for p in model.parameters():
    total_param.append(int(p.numel()))
    
sum(total_param)

len(encoded_text)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
criterion = nn.CrossEntropyLoss()

# Training Data and Validation Data
# percentage of data to be used for training
train_percent = 0.1

len(encoded_text)

int(len(encoded_text) * (train_percent))

train_ind = int(len(encoded_text) * (train_percent))

train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]

# Params

# Epochs to train for
epochs = 50
# batch size 
batch_size = 128

# Length of sequence
seq_len = 100

# for printing report purposes
# always start at 0
tracker = 0

# number of characters in text
num_char = max(encoded_text)+1

# Training Add

# Set model to train
model.train()


# Check to see if using GPU
if model.use_gpu:
    model.cuda()

for i in range(epochs):
    
    hidden = model.hidden_state(batch_size)
    
    
    for x,y in generate_batches(train_data,batch_size,seq_len):
        
        tracker += 1
        
        # One Hot Encode incoming data
        x = one_hot_encoder(x,num_char)
        
        # Convert Numpy Arrays to Tensor
        
        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y)
        
        # Adjust for GPU if necessary
        
        if model.use_gpu:
            
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        # Reset Hidden State
        # If we dont' reset we would backpropagate through all training history
        hidden = tuple([state.data for state in hidden])
        
        model.zero_grad()
        
        lstm_output, hidden = model.forward(inputs,hidden)
        loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
        loss.backward()
        
        # POSSIBLE EXPLODING GRADIENT PROBLEM!
        # LET"S CLIP JUST IN CASE
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
        
        optimizer.step()
        
        
        
        ###################################
        ### CHECK ON VALIDATION SET ######
        #################################
        
        if tracker % 25 == 0:
            
            val_hidden = model.hidden_state(batch_size)
            val_losses = []
            model.eval()
            
            for x,y in generate_batches(val_data,batch_size,seq_len):
                
                # One Hot Encode incoming data
                x = one_hot_encoder(x,num_char)
                

                # Convert Numpy Arrays to Tensor

                inputs = torch.from_numpy(x)
                targets = torch.from_numpy(y)

                # Adjust for GPU if necessary

                if model.use_gpu:

                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    
                # Reset Hidden State
                # If we dont' reset we would backpropagate through 
                # all training history
                val_hidden = tuple([state.data for state in val_hidden])
                
                lstm_output, val_hidden = model.forward(inputs,val_hidden)
                val_loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
                val_losses.append(val_loss.item())
            
            # Reset to training model after val for loop
            model.train()
            
            print(f"Epoch: {i} Step: {tracker} Val Loss: {val_loss.item()}")

# Save Model

model_name = 'example.net'

torch.save(model.state_dict(),model_name)

# Load Model
# MUST MATCH THE EXACT SAME SETTINGS AS MODEL USED DURING TRAINING!

model = CharModel(
    all_chars = all_characters,
    num_hidden = 512,
    num_layers = 3,
    drop_prob = 0.5,
    use_gpu = True,
)
model.load_state_dict(torch.load(model_name))
model.eval()


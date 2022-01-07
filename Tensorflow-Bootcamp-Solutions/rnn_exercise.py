# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 01:50:49 2019

@author: tanma
"""

import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("monthly-milk-production.csv",index_col = "Month")

train_set = data.head(156)
test_set = data.tail(12)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)

def next_batch(training_data,steps,batch_size):
    rand_start = np.random.randint(0,len(training_data)-steps)
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    
    return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)

num_inputs = 1
steps = 12
neurons = 100
num_outputs = 1
learning_rate = 0.03
iterations = 4000
batch_size = 1

x = tf.placeholder(tf.float32,[None,steps,num_inputs])
y = tf.placeholder(tf.float32,[None,steps,num_outputs])
hold_prob = tf.placeholder(tf.float32)

cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units = neurons,activation = tf.nn.leaky_relu)

op = tf.contrib.rnn.OutputProjectionWrapper(cell_1,output_size = num_outputs)

ops,states = tf.nn.dynamic_rnn(op,x,dtype = tf.float32)

loss = tf.reduce_mean(tf.square(ops - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(iterations):
        batch_x,batch_y = next_batch(train_set,steps,batch_size)
        
        sess.run(train,feed_dict={x:batch_x,y:batch_y})
        
        if i%100 == 0:
            
            mse = loss.eval(feed_dict={x:batch_x,y:batch_y})
            print(i,'\tMSE:',mse)
    
    
    saver.save(sess, "./ex_time_series_model")      

with tf.Session() as sess:
    saver.restore(sess,"./ex_time_series_model")
    
    train_seed = list(train_set[-12:])
    
    for iteration in range(12):
        x_batch = np.array(train_seed[-steps:]).reshape(-1,steps,1)
        
        y_pred = sess.run(ops,feed_dict={x:x_batch})
        
        train_seed.append(y_pred[0,-1,0])
        
train_seed = train_seed[-12:]        
train_seed = scaler.inverse_transform(np.array(train_seed).reshape(12,1))
test_set["Generated"] = train_seed       
print(test_set)
    
    
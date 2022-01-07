# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("cal_housing_clean.csv")
y = data["medianHouseValue"]
x = data.drop("medianHouseValue",axis = 1)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = pd.DataFrame(data = scaler.transform(x_train),columns = x_train.columns,index = x_train.index)
x_test = pd.DataFrame(data = scaler.transform(x_test),columns = x_test.columns,index = x_test.index)

age = tf.feature_column.numeric_column("housingMedianAge")
rooms = tf.feature_column.numeric_column("totalRooms")
bedrooms = tf.feature_column.numeric_column("totalBedrooms")
population = tf.feature_column.numeric_column("population")
households = tf.feature_column.numeric_column("households")
med_inc = tf.feature_column.numeric_column("medianIncome")

feat_cols = [age,rooms,bedrooms,population,households,med_inc]

input_func = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train,batch_size = 10,num_epochs = 2000,shuffle = True)
estimator = tf.estimator.DNNRegressor(hidden_units = [8,8,8],feature_columns = feat_cols)

estimator.train(input_fn = input_func,steps = 2000)

predict_input_fn = tf.estimator.inputs.pandas_input_fn(x = x_test,shuffle = False)
preds = estimator.predict(predict_input_fn)
preds = list(preds)

final_preds = []

for pred in preds:
    final_preds.append(pred["predictions"])

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,final_preds)**0.5)
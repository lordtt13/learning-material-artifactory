# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:59:08 2019

@author: tanma
"""

import tensorflow as tf
import pandas as pd

data = pd.read_csv("census_data.csv")

def label_fix(label):
    if label == " <=50K":
        return 0
    else:
        return 1
    
data["income_bracket"] = data["income_bracket"].apply(label_fix)

x = data.drop("income_bracket",axis = 1)
y = data["income_bracket"]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

age = tf.feature_column.numeric_column("age")
workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass',data["workclass"].unique())
education = tf.feature_column.categorical_column_with_vocabulary_list('education',data["education"].unique())
education_num = tf.feature_column.numeric_column("education_num")
marital_status = tf.feature_column.categorical_column_with_vocabulary_list('marital_status',data["marital_status"].unique())
occupation = tf.feature_column.categorical_column_with_vocabulary_list('occupation',data["occupation"].unique())
relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship',data["relationship"].unique())
race = tf.feature_column.categorical_column_with_vocabulary_list('race',data["race"].unique())
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender',data["gender"].unique())
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")
native_country = tf.feature_column.categorical_column_with_vocabulary_list('native_country',data["native_country"].unique())

feat_cols = [age,workclass,education,education_num,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,native_country]

input_func = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train,batch_size = 10,num_epochs = 2000,shuffle = True)

model = tf.estimator.LinearClassifier(feature_columns = feat_cols)
model.train(input_fn = input_func,steps = 5000)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test,batch_size = len(x_test),shuffle = False)
preds = model.predict(pred_input_func)
preds = list(preds)
preds = [pred["class_ids"] for pred in preds]

from sklearn.metrics import classification_report
print(classification_report(y_test,preds,[0,1]))
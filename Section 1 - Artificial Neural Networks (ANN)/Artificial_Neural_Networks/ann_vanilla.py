# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:30:07 2020

@author: Tanmay Thakur
"""

import numpy as np

x=np.array([[4.85, 9.63], [8.62, 3.23], [5.43, 8.23], [9.21, 6.34]])
y=np.array([[1], [0], [1], [0]])

def sigmoid(z):
    return 1/(np.exp(-z)+1)



theta_1 = 0
theta_2 = 0
error_1 = 1 # dummy_value
error_2 = 1 # dummy_value
lr = 0.1
tol = 0.1

while(abs(float(error_1)) and abs(float(error_2)) > tol):
    sum1=0
    sum2=0
    for j in range(4):
        z = theta_1*x[j][0]+theta_2*x[j][1]

        h = sigmoid(z)

        error_1 = (h-y[j])*x[j][0]

        sum1 += error_1
        error_2=(h-y[j])*x[j][1]

        sum2 += error_2
        
    print('values of h:', h)
    print('value of z:', z)
    theta_1 -= (lr*sum1)
    theta_2 -= (lr*sum2)
    print('theta1: ',theta_1)
    print('theta2:', theta_2)
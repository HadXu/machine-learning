# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 2, 2, 2

# Create random input and output data
x = np.array([[1],[2]])
y = np.array([[-1],[-2]])

# Randomly initialize weights
w1 = np.zeros((D_in,H))
w2 = np.ones((H,D_out))

def sigmoid(x):
    return 1./(1+np.exp(-x))

def dev_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

learning_rate = 0.5
for t in range(100000):
    z1 = np.dot(w1,x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2,a1)
    a2 = z2

    loss = a2 - y

    # print(np.square(loss).sum())

    dz2 = a2 - y
    dw2 = np.dot(dz2,a1.T)
    dz1 = np.dot(w2.T,dz2) * dev_sigmoid(z1)
    dw1 = np.dot(dz1,x.T)


    # Update weights
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2


z1  = np.dot(w1,x)
a1 = sigmoid(z1)
z2 = np.dot(w2,a1)
a2 = z2

print(a2)





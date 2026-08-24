'''Implement a 2-layer (input layer, hidden
layer and output layer) neural network from
scratch for the XOR operation. This includes
implementing forward and backward passes from scratch.
The truth table for XOR is given by
'''

# x1 | x2 | y(x1 XOR x2)
# ---------------------
# 0  | 1  |     0
# 0  | 1  |     1
# 1  | 0  |     1
# 1  | 1  |     0

import numpy as np
# XOR dataset
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
],dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
],dtype=float)

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

#initialize weights and biases
np.random.seed(123)

# input layer to Hidden layer
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

# hidden layer to output layer
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

# training parameters
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):

    # forward pass
    #hidden layer
    z1 = np.dot(x,W1) + b1
    a1 = sigmoid(z1)

    # output layer
    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)

    # backward pass
    # output layer error
    error = y - a2

    # gradient at output layer
    delta = error * sigmoidDeriv(a2)

    # gradient for w2 and b2
    dW2 = np.dot(a1.T, delta)
    db2 = np.sum(delta, axis=0, keepdims=True)

    # hidden layer error
    delta1 = np.dot(delta, W2.T)

    # Gradient for W1 and b1
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # update weights
    W2 += learning_rate * dW2
    b2 += learning_rate * db2

    W1 += learning_rate * dW1
    b1 += learning_rate * db1

    # print loss occasionally
    if epoch % 100 == 0:
        loss = np.mean((y-a2)**2)
        print(f"epoch: {epoch}, loss: {loss}")

# Testing
print("XOR Predictions: ")
z1 = np.dot(x,W1) + b1
a1 = sigmoid(z1)

z2 = np.dot(a1,W2) + b2
predictions = sigmoid(z2)

for i in range(len(x)):
    print(
        f"{x[i]} --> "
        f" Prediction: {predictions[i][0]:.4f}"
        f" Class: {round(predictions[i][0])}"
        f" Actual: {int(y[i][0])}"
    )
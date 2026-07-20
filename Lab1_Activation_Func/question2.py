'''Write down the observations from the plot for all the above functions in the code.
    a) What are the min and max values for the functions?
    b) Is the output of the function zero-centred?
    c) What happens to the gradient when the input values are too small or too big?
    d) What is the relationship between sigmoid and tanh?
'''

from matplotlib import pyplot as plt
import numpy as np

z = np.linspace(-10, 10, 100)
# print(z)

# Sigmoid Function
# f(x) = 1/1+e^-x
print("a) What are the min and max values for the functions?")
def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

sigval = sigmoid(z)
print(f"min: {min(sigval)}, max: {max(sigval)} --> Sigmoid value")

def tanh(x):
    posexp = np.exp(x)
    negexp = np.exp(-x)
    return (posexp - negexp)/(posexp + negexp)

tanhval = tanh(z)
print(f"min: {min(tanhval)}, max: {max(tanhval)} -->Tanh value")

def relu(x):
    return np.maximum(0, x)

reluval = relu(z)
print(f"min: {min(reluval)}, max: {max(reluval)} -->ReLU value")

alpha = 0.01
def leaky_relu(x):
    return np.maximum( x, alpha * x)
leakyval = leaky_relu(z)
print(f"min: {min(leakyval)}, max: {max(leakyval)} -->Leaky ReLU value")

#b) Is the output of the function zero-centred?
# Sigmoid: No
#  output lies between 0 and 1

# Thanh: Yes
# output lies between -1 and 1 and is symmetric about zero.

# ReLU: No
# output are either 0 or pos.

#Leaky ReLU
# produces small negative outputs for negative inputs.

#c) What happens to the gradient when the input values are too small or too big?

# Sigmoid:
# The gradient becomes nearly zero for very large positive or negative input

# Tanh:
# The gradient also approaches zero at both extremes (vanishing gradient).

# ReLU:
# Gradient is 0 for negative inputs and 1 for positive inputs.
# so Neurons may become inactive for negative inputs (dying ReLU).

# Leaky ReLU:
# Gradient is alpha (0.01) for negative inputs and 1 for positive inputs.
# It avoids the dying ReLU problem by allowing a small gradient.

#d) Relationship between Sigmoid and Tanh

# Both are S- shaped activation functions.
# Tanh is a scaled and shifted version of the sigmoid function.
# tanh(x) = 2sigmoid(2x) -1
# sigmoid output values between 0 and 1.
# Tanh outputs values between -1 and 1 and is zero-centered, making it generally preferable for hidden layers.
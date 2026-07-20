'''Implement the following functions in Python from scratch.
Do not use any library functions. You are allowed to use numpy and matplotlib.
Generate 100 equally spaced values between -10 and 10. Call this list as  z.
Implement the following functions and its derivative. Use class notes to find
the expression for these functions. Use z as input and plot both the function
outputs and its derivative outputs.
    Sigmoid
    Tanh
    ReLU (Rectified Linear Unit)
    Leaky ReLU
    Softmax (no need for visualization)
'''

from matplotlib import pyplot as plt
import numpy as np

z = np.linspace(-10, 10, 100)
# print(z)

# Sigmoid Function
# f(x) = 1/1+e^-x

def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

def derivative_sigmoid(x):
    s = sigmoid(x)
    derivative = s * (1 - s)
    return derivative

# Tanh Function
# f(x) = (e^x -e^-x) / (e^x + e^-x)
def tanh(x):
    posexp = np.exp(x)
    negexp = np.exp(-x)
    return ((posexp - negexp)/(posexp + negexp))

def tanh_derivative(x):
    th = tanh(x)
    derivative = 1.0 - th
    return derivative


# ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum( x, 0)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Leaky ReLU
alpha = 0.01
def leaky_relu(x):
    return np.maximum( x, alpha * x)
def leaky_relu_derivative(x):
    return np.where(x>0, 1, alpha)

# softmax
def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)

sofmaxoutput = softmax(z)
print(sofmaxoutput)

plt.figure(figsize=(10, 10))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid')
plt.plot(z, derivative_sigmoid(z), label='Sigmoid Derivative')
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()

# Tanh
plt.subplot(2, 2, 2)
plt.plot(z, tanh(z), label='Tanh')
plt.plot(z, tanh_derivative(z), label='Tanh Derivative')
plt.title("Tanh")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()

# ReLU
plt.subplot(2, 2, 3)
plt.plot(z, relu(z), label='ReLU')
plt.plot(z, relu_derivative(z),label='ReLU Derivative')
plt.title("ReLU")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(z, leaky_relu(z), label='Leaky ReLU')
plt.plot(z, leaky_relu_derivative(z), label='Leaky ReLU Derivative')
plt.title("Leaky ReLU")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()

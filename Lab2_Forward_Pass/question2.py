# Implement forward pass for the above two networks.
# Print activation values for each neuron at each layer.
# Print the loss value (y^).
# Implement the forward pass using vectorized operations,
# i.e. W should be a matrix, x, z and a are vectors.
# The implementation should not contain any loops.
from array import array

# Random weight generated

import numpy as np

# calculate z = sum(WiXi +b)
def calcZ(x,w,b):
    return np.dot(x,w)+b

# calculate a with different activation function
# sigmoid activation
def calcAsig(z):
    A = 1/(1+np.exp(-z))
    return A
# tanH activation
def calcAtanH(z):
    posexp = np.exp(z)
    negexp = np.exp(-z)
    return (posexp-negexp)/(posexp+negexp)
# relu activation
def calcArelu(z):
    return np.maximum(0,z)

def mainloop(x,w,b,activation):
    loop = 1
    for val,bias in zip(w,b):
        z = calcZ(x,val,bias)
        print(f"z values for Layer {loop} : {z} ")
        if activation == "sigmoid":
            a = calcAsig(z)
        elif activation == "tanh":
            a = calcAtanH(z)
        elif activation == "relu":
            a = calcArelu(z)

        print(f"a values with {activation} activation for Layer {loop}: {a}")

        x = a
        loop+=1
    print("done")
    print(f"Loss value (y^): {a}")
    print("\n")

x = np.array(input("Enter the values of X separated by space: ").split(), dtype=float)
n = int(input("How many layers do you want? "))
m = len(x)
w = []
b = []
for i in range(n):
    ni = int(input(f"How many neurons do you want in layer {i+1}? "))
    bi = float(input(f"What should be the bias of layer {i+1}? "))

    wi = np.random.rand(m,ni)
    bi = np.full(ni,bi)
    w.append(wi)
    b.append(bi)
    m = ni

print("Weights:")
for layer in w:
    print(layer)

mainloop(x,w,b,"sigmoid")
mainloop(x,w,b,"tanh")
mainloop(x,w,b,"relu")




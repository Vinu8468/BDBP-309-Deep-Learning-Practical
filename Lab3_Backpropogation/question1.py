# Implement backward pass for the above two
# networks. Print the gradient values for
# each neuron in each layer.

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

# Derivatives of different activation function
def dsigmoid(z):
    s = calcAsig(z)
    return s*(1-s)

def dtanh(z):
    t = calcAtanH(z)
    return 1 - t**2

def drelu(z):
    return (z>0).astype(float)

# forward pass

def mainloop(x,w,b,activation):
    loop = 1
    for wi,bi in zip(w,b):
        z = calcZ(x,wi,bi)
        print(f"Layer {loop}: z={z} ")
        if activation == "sigmoid":
            a = calcAsig(z)
        elif activation == "tanh":
            a = calcAtanH(z)
        elif activation == "relu":
            a = calcArelu(z)

        print(f"a =",a)

        x = a
        loop+=1
    print("Prediction (y) =", a)
    print("\n")

# forward pass (stores values)
def forward(x,w,b,activation):
    activations = [x]
    z_values = []
    for wi, bi in zip (w,b):
        z  = calcZ(x,wi,bi)
        z_values.append(z)

        if activation == "sigmoid":
            a = calcAsig(z)
        elif activation == "tanh":
            a = calcAtanH(z)
        elif activation == "relu":
            a = calcArelu(z)
        activations.append(a)
        x = a
    return activations, z_values


# Backpropagation
def backprop(w,activations,z_values,y,activation):
    gradW = []
    gradB = []
    y_hat = activations[-1]

    # Output layer delta
    if activation == "sigmoid":
        delta = (y_hat-y) * dsigmoid(z_values[-1])
    elif activation == "tanh":
        delta = (y_hat-y) * dtanh(z_values[-1])
    elif activation == "relu":
        delta = (y_hat-y) * drelu(z_values[-1])

    # Last layer gradients
    dW = np.outer(activations[-2],delta)
    dB = delta

    gradW.insert(0, dW)
    gradB.insert(0, dB)
    # Hidden layers
    for i in range(len(w)-2,-1,-1):
        if activation == "sigmoid":
            delta = np.dot(delta, w[i + 1].T) * dsigmoid(z_values[i])
        elif activation == "tanh":
            delta = np.dot(delta, w[i + 1].T) * dtanh(z_values[i])
        elif activation == "relu":
            delta = np.dot(delta, w[i + 1].T) * drelu(z_values[i])
        dB = delta

        gradW.insert(0, dW)
        gradB.insert(0, dB)
    # print gradients
    print("backprop")

    for i in range(len(gradW)):
        print(f"Layer {i+1}")
        print("Gradient of Weights: ")
        print(gradW[i])

        print("Gradient of Bias: ")
        print(gradB[i])
    print("="*30)



# Main Program
x = np.array(
    input("Enter input values separated by space:").strip().split(),
    dtype=float
)

n = int(input("Number of layers: "))

m = len(x)
w = []
b = []
for i in range(n):
    neurons = int(input(f"Enter number of neurons {i+1}: "))
    bias = float(input(f"Enter bias value {i+1}: "))
    wi = np.random.rand(m,neurons)
    bi = np.full(neurons,bias)
    w.append(wi)
    b.append(bi)
    m = neurons

print("Random Weights")
for i, layer in enumerate(w):
    print(f"Layer {i+1}")
    print(layer)
    print()

# sigmoid
print("Sigmoid")
mainloop(x,w,b,activation="sigmoid")
A, Z = forward(x,w,b,activation="sigmoid")
y = np.ones_like(A[-1]) # dummy target

backprop(w,A,Z,y ,activation="sigmoid")

#TanH
print("TanH")
mainloop(x,w,b,activation="tanh")
A, Z = forward(x,w,b,activation="tanh")
y = np.ones_like(A[-1])

backprop(w,A,Z,y ,activation="tanh")

# ReLu
print("ReLU")
mainloop(x,w,b,activation="relu")
A,Z = forward(x,w,b,activation="relu")
y = np.ones_like(A[-1])
backprop(w,A,Z,y ,activation="relu")

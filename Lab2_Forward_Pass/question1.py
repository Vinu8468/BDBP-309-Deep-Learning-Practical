# Consider the following two networks.  W is a matrix, x is a vector,
# z is a vector, and a is a vector. y^ is a scalar and a final prediction.
# Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
# Initialize X and W randomly. Every neuron has a bias term.

import numpy as np

# calculate z = sum(WiXi +b)
def calcZ(x,w):
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

def mainloop(x,w,activation):
    loop = 1
    for val in w:
        z = calcZ(x,val)
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

x = [0.3,-1.2]
w = [[[0.1,-0.1,0.2],[-1.1,0.4,1.1]],[[0.2,0.3],[0.1,-0.1],[-0.2,-0.1]],[0.3,-0.3]]
b = 0.01

mainloop(x,w,activation="sigmoid")
mainloop(x,w,activation="tanh")
mainloop(x,w,activation="relu")


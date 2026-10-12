# Implement a 1-layer (input - output layer) neural network from
# scratch for the following dataset. This includes implementing
# forward and backward passes from scratch. Print the training
# loss and plot it over 1000 iterations.

import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,0,1]
],dtype=np.float32)

y = np.array([[0],[1],[1],[0]],dtype=np.float32)

# initialize weights
np.random.seed(42)
W = np.random.randn(3,1)*0.1
learing_rate = 0.1
iterations = 1000

losses = []
weights_history = []
gradient_history = []

# sigmoid activation
def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

# training loop
for i in range(iterations):
    # forward propagation
    z = X @ W
    predictions = sigmoid(z)

    # binary cross entropy loss
    eps = 1e-12
    loss = -np.mean(
        y * np.log(predictions+eps)
        + (1-y)*np.log(1-predictions+eps)
    )
    # backpropagation
    gradients = X.T @ (predictions -y)/len(y)

    # update weights
    W = W - learing_rate * gradients
    # values for visualisation
    losses.append(loss)
    weights_history.append(W.flatten().copy())
    gradient_history.append(gradients.flatten().copy())

# print results
print("Final Weights:")
print(W)

print("Final predictions:")
print(np.round(sigmoid(X@W),3))

print("final training loss:", round(losses[-1],5))

# plot training loss
plt.figure(figsize=(10,5))
plt.title("Training Loss over 1000 Iterations")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.plot(range(1,iterations+1),losses)
plt.grid(True)
plt.show()

# visualize weights
weights_history = np.array(weights_history)

plt.figure(figsize=(10,5))
for j in range(3):
    plt.plot(weights_history[:,j], label =f"Weight W{j+1}")
plt.xlabel("Iterations")
plt.ylabel("Weight Value")
plt.title("Weights during Training")
plt.legend()
plt.grid(True)
plt.show()

# visual gradients
gradiateds_history = np.array(gradient_history)

plt.figure(figsize=(10,5))
for j in range(3):
    plt.plot(gradiateds_history[:,j], label =f"Gradient W{j+1}")

plt.xlabel("Iterations")
plt.ylabel("Gradient Value")
plt.title("Gradients during Training")
plt.legend()
plt.grid(True)
plt.show()

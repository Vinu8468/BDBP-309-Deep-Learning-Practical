# Write a program to simulate vanishing
# & exploding gradient problems.
# Plot the gradient values to demonstrate the issues with training a very deep network.

import numpy as np
import matplotlib.pyplot as plt

# number of layers
layers = np.arange(1,51)

# Gradient multipliers
vanishing_factor = 0.5
exploding_factor = 1.5

# Calculate gradients
vanishing_gradient = vanishing_factor ** layers
exploding_gradient = exploding_factor ** layers

# # plot vanishing gradient
# plt.figure(figsize=(10,10))
# plt.plot(layers, vanishing_gradient,marker='o')
# plt.xlabel("Number of layers")
# plt.ylabel("Gradient Value")
# plt.title("Vanishing Gradient Problem")
# plt.grid(True)
# plt.show()
#
# # Plot exploding gradient
# plt.figure(figsize=(10,10))
# plt.plot(layers, exploding_gradient,marker='o')
# plt.xlabel("Number of layers")
# plt.ylabel("Gradient Value")
# plt.title("Exploding Gradient Problem")
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 10))

plt.plot(layers,vanishing_gradient,marker='o',label='Vanishing Gradient')

plt.plot(layers,exploding_gradient,marker='o',label='Exploding Gradient')

plt.xlabel("Number of Layers")
plt.ylabel("Gradient Value")
plt.title("Vanishing and Exploding Gradient Problems")

plt.yscale("log")
plt.legend()
plt.grid(True)

plt.show()
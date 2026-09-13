"""
Day 11 — Neural Network Fundamentals
A single-neuron implementation using Python + NumPy.
"""

import numpy as np


def neuron(inputs, weights, bias):
    inputs = np.asarray(inputs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    z = np.dot(inputs, weights) + bias
    return z


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = [2.0, 3.0]
    w = [0.5, -0.2]
    b = 0.1

    z = neuron(x, w, b)
    print("Weighted sum:", z)
    print("Sigmoid output:", sigmoid(z))

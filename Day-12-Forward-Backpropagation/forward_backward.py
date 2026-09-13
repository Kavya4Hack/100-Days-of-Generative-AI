"""
Day 12 — Forward Propagation & Backpropagation
A tiny one-neuron example showing analytical gradient calculation.
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_example():
    x = 2.0
    y = 1.0
    w = 0.5
    b = 0.0

    # Forward pass
    z = w * x + b
    prediction = sigmoid(z)

    # Binary cross-entropy loss
    eps = 1e-8
    loss = -(y * np.log(prediction + eps) +
             (1 - y) * np.log(1 - prediction + eps))

    # For sigmoid + binary cross entropy:
    # dL/dz = prediction - y
    dz = prediction - y
    dw = dz * x
    db = dz

    print("Prediction:", prediction)
    print("Loss:", loss)
    print("dL/dw:", dw)
    print("dL/db:", db)


if __name__ == "__main__":
    run_example()

"""
Day 13 — Gradient Descent
Fit y = wx + b using manual gradient descent.
"""

import numpy as np


def train(x, y, learning_rate=0.01, epochs=1000):
    w, b = 0.0, 0.0
    n = len(x)

    for epoch in range(epochs):
        predictions = w * x + b
        error = predictions - y

        loss = np.mean(error ** 2)

        dw = (2 / n) * np.sum(error * x)
        db = (2 / n) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

    return w, b


if __name__ == "__main__":
    x = np.array([1., 2., 3., 4., 5.])
    y = 4 * x - 3

    w, b = train(x, y)
    print(f"\nLearned equation: y = {w:.4f}x + {b:.4f}")

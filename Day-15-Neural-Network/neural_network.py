"""
Day 15 — Build a Neural Network with PyTorch
"""

import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = SimpleNetwork()
    x = torch.tensor([[1.0], [2.0], [3.0]])

    predictions = model(x)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:\n", predictions)

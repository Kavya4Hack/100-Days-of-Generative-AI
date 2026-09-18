import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Fully connected neural network for Fashion-MNIST classification."""

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        x = x.view(x.size(0), -1)

        # [batch_size, 784] -> [batch_size, 128]
        x = self.relu(self.fc1(x))

        # [batch_size, 128] -> [batch_size, 64]
        x = self.relu(self.fc2(x))

        # [batch_size, 64] -> [batch_size, 10]
        x = self.fc3(x)

        return x

"""
Day 18 — CNN, RNN, LSTM & GRU
Small PyTorch examples showing the expected input/output shapes.
"""

import torch
import torch.nn as nn


class CNNExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.layer(x)


class RNNExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.RNN(input_size=4, hidden_size=8, batch_first=True)

    def forward(self, x):
        return self.layer(x)


class LSTMExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)

    def forward(self, x):
        return self.layer(x)


class GRUExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.GRU(input_size=4, hidden_size=8, batch_first=True)

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    image_batch = torch.randn(2, 3, 32, 32)
    sequence_batch = torch.randn(2, 10, 4)

    print("CNN output:", CNNExample()(image_batch).shape)
    print("RNN output:", RNNExample()(sequence_batch)[0].shape)
    print("LSTM output:", LSTMExample()(sequence_batch)[0].shape)
    print("GRU output:", GRUExample()(sequence_batch)[0].shape)

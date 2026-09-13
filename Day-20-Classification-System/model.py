"""
Day 20 — Neural Network Classification System
Model definition.
"""

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.network(x)

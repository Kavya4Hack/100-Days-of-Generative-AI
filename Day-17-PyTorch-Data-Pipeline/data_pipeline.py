"""
Day 17 — Dataset, DataLoader, batching, epochs and learning rate.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader


X = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
)

y = torch.tensor(
    [[1.0], [5.0], [9.0], [13.0], [17.0], [21.0]]
)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")

    for batch_x, batch_y in loader:
        print("Batch X:", batch_x.flatten().tolist())
        print("Batch y:", batch_y.flatten().tolist())

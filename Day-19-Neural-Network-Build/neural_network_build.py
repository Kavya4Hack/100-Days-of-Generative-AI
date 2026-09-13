"""
Day 19 — Neural Network Build
A complete small regression network with training and loss monitoring.
"""

import torch
import torch.nn as nn


torch.manual_seed(42)

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[1.0], [5.0], [9.0], [13.0], [17.0]])


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


model = Network()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()

    predictions = model(X)
    loss = loss_function(predictions, y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

print("\nPredictions:")
print(model(X).detach())

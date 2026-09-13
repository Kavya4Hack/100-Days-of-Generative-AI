"""
Day 16 — PyTorch Training Loop

Learns the relationship:
y = 4x - 3
"""

import torch
import torch.nn as nn


torch.manual_seed(42)

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[1.0], [5.0], [9.0], [13.0], [17.0]])

model = nn.Linear(1, 1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(X)
    loss = loss_function(predictions, y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

print("\nLearned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

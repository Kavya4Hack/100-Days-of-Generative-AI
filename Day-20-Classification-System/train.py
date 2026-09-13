"""
Day 20 — Train / Validate / Save

Dataset: Iris
Model: Small feed-forward neural network
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import Classifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("classifier.pt")


def accuracy(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            predictions = model(X).argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    data = load_iris()

    X_train, X_temp, y_train, y_temp = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=42,
        stratify=data.target,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = Classifier(input_features=4, num_classes=3).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            train_acc = accuracy(model, train_loader)
            val_acc = accuracy(model, val_loader)

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.3f} | "
                f"Val Acc {val_acc:.3f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 4,
            "num_classes": 3,
        },
        MODEL_PATH,
    )

    print(f"\nSaved model to: {MODEL_PATH.resolve()}")
    print(f"Test accuracy: {accuracy(model, test_loader):.3f}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.optim import Adam

from model import NeuralNetwork


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train the model and return training history."""

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(predictions, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()

        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images)
                loss = loss_function(predictions, labels)

                total_val_loss += loss.item()

                _, predicted = torch.max(predictions, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

    return history

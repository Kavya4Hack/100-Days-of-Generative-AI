import torch


def evaluate_model(model, data_loader):
    """Evaluate classification accuracy on a dataset."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

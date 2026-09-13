"""
Day 20 — Model Evaluation
Loads the saved classifier and evaluates it on the test split.
"""

from pathlib import Path

import torch
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import Classifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("classifier.pt")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "classifier.pt not found. Run train.py first."
        )

    data = load_iris()

    _, X_temp, _, y_temp = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=42,
        stratify=data.target,
    )

    _, X_test, _, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = Classifier(
        checkpoint["input_features"],
        checkpoint["num_classes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(DEVICE))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.tolist())

    print("Confusion Matrix:")
    print(confusion_matrix(targets, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            targets,
            predictions,
            target_names=data.target_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()

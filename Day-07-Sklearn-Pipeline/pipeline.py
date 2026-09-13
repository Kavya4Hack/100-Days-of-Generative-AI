"""
Day 07 — Scikit-learn Pipeline
End-to-end preprocessing + model training.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def build_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    joblib.dump(pipeline, "iris_pipeline.joblib")
    print("Saved: iris_pipeline.joblib")


if __name__ == "__main__":
    main()

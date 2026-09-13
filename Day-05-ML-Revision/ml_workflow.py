"""
Day 05 — Machine Learning Revision
Demonstrates supervised vs. unsupervised learning workflows.
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def supervised_example():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Supervised accuracy:", accuracy_score(y_test, predictions))


def unsupervised_example():
    data = load_iris()
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(data.data)

    print("Generated cluster labels:", labels[:10])


if __name__ == "__main__":
    supervised_example()
    unsupervised_example()

"""
Day 04 — NumPy & Pandas
Small numerical and tabular-data practice.
"""

import numpy as np
import pandas as pd


def main():
    scores = np.array([72, 85, 91, 64, 88], dtype=float)

    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("Scores + 5:", scores + 5)

    df = pd.DataFrame(
        {
            "student": ["A", "B", "C", "D", "E"],
            "score": [72, 85, np.nan, 64, 88],
            "department": ["AI", "AI", "ML", "ML", "AI"],
        }
    )

    df["score"] = df["score"].fillna(df["score"].mean())

    print("\nCleaned data:")
    print(df)

    print("\nAverage score by department:")
    print(df.groupby("department")["score"].mean())

    print("\nStudents scoring >= 80:")
    print(df[df["score"] >= 80])


if __name__ == "__main__":
    main()

"""
Day 10 — AI Data Analyst v1

A simple architecture:
CSV → Pandas → analysis → natural-language report.

The LLM layer is intentionally abstract so the data-processing logic remains
independent from a specific provider.
"""

from pathlib import Path
import pandas as pd


def analyze_csv(path):
    df = pd.read_csv(path)

    report = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": df.describe(numeric_only=True).to_dict(),
    }

    return df, report


def format_report(report):
    lines = [
        f"Rows: {report['rows']}",
        f"Columns: {', '.join(report['columns'])}",
        f"Missing values: {report['missing_values']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sample = Path("sample_data.csv")

    if sample.exists():
        _, report = analyze_csv(sample)
        print(format_report(report))
    else:
        print("Create sample_data.csv in this folder to run the analysis.")

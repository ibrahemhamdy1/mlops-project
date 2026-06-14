"""Data validation gate — bad data never reaches training.

Two layers:
  1. pandera DataFrameSchema — typed, declarative column contracts
  2. custom checks pandera doesn't express (volume, target domain)

Production equivalent of the same idea: Great Expectations / Glue DQ.
Exits non-zero on any violation so the CI job fails.
"""
import sys

import pandas as pd
import pandera as pa
from sklearn.datasets import load_breast_cancer

EXPECTED_FEATURES = 30
MIN_ROWS = 100


def load_dataset() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


def build_schema(df: pd.DataFrame) -> pa.DataFrameSchema:
    """Every feature: float, non-null, non-negative (physical measurements)."""
    feature_cols = [c for c in df.columns if c != "target"]
    columns = {
        col: pa.Column(float, checks=pa.Check.ge(0), nullable=False)
        for col in feature_cols
    }
    columns["target"] = pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(columns, strict=True)


def validate(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    feature_cols = [c for c in df.columns if c != "target"]
    if len(feature_cols) != EXPECTED_FEATURES:
        errors.append(
            f"schema: expected {EXPECTED_FEATURES} features, got {len(feature_cols)}"
        )

    try:
        build_schema(df).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.head(20).iterrows():
            errors.append(f"pandera: column={row['column']} check={row['check']}")

    if len(df) < MIN_ROWS:
        errors.append(f"volume: only {len(df)} rows — below minimum {MIN_ROWS}")

    return errors


def main() -> int:
    df = load_dataset()
    errors = validate(df)
    if errors:
        print("DATA VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Data validation passed: {len(df)} rows, {EXPECTED_FEATURES} features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

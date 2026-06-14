"""Evidently drift report — the industry-standard drift tool.

Generates a full HTML data-drift report (per-feature tests, distributions)
comparing current inference inputs against the training reference data.
Uploaded as a CI artifact by the drift workflow.

drift_check.py stays as the fast PSI gate (exit code drives retraining);
this produces the human-readable investigation report.
"""
import argparse
import json
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.datasets import load_breast_cancer


def load_current(simulate_drift: bool) -> pd.DataFrame:
    x = load_breast_cancer(as_frame=True).data
    if simulate_drift:
        x = x * 1.5
    return x


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="model/reference_data.csv")
    p.add_argument("--out", default="drift_report.html")
    p.add_argument("--simulate-drift", action="store_true")
    args = p.parse_args()

    reference = pd.read_csv(args.reference)
    current = load_current(args.simulate_drift)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(args.out)

    summary = report.as_dict()["metrics"][0]["result"]
    print(
        json.dumps(
            {
                "drift_detected": summary["dataset_drift"],
                "drifted_features": summary["number_of_drifted_columns"],
                "total_features": summary["number_of_columns"],
                "report": args.out,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

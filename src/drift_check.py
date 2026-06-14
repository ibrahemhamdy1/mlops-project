"""Drift detection via PSI (Population Stability Index).

Compares current feature distributions against the reference stats
captured at training time. PSI rule of thumb:
  < 0.1   no drift
  0.1-0.2 moderate — watch
  > 0.2   significant — retrain trigger

Run by the scheduled drift workflow. Exit code 2 = drift detected
(the workflow uses that to trigger retraining).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

PSI_ALERT = 0.2
EPS = 1e-6


def psi(expected_ratios: list[float], actual: np.ndarray, bin_edges: list[float]) -> float:
    counts, _ = np.histogram(actual, bins=np.array(bin_edges))
    actual_ratios = counts / max(counts.sum(), 1)
    e = np.clip(np.array(expected_ratios), EPS, None)
    a = np.clip(actual_ratios, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_current(simulate_drift: bool) -> pd.DataFrame:
    """In production this reads recent inference inputs from your
    feature log / S3. Here we use fresh dataset rows, optionally
    perturbed to demonstrate a drift alert."""
    x = load_breast_cancer(as_frame=True).data
    if simulate_drift:
        x = x * 1.5  # shift every distribution -> guaranteed PSI alert
    return x


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="model/reference_stats.json")
    p.add_argument("--simulate-drift", action="store_true")
    args = p.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    current = load_current(args.simulate_drift)

    drifted = {}
    for col, stats in ref.items():
        score = psi(stats["ratios"], current[col].to_numpy(), stats["bin_edges"])
        if score > PSI_ALERT:
            drifted[col] = round(score, 3)

    if drifted:
        print(f"DRIFT DETECTED on {len(drifted)} feature(s) (PSI > {PSI_ALERT}):")
        for col, score in sorted(drifted.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {col}: PSI={score}")
        return 2

    print(f"No drift: all features PSI <= {PSI_ALERT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

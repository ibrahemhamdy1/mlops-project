"""Evaluation gate — the model equivalent of 'tests must pass'.

Rules:
  1. New AUC must clear an absolute floor (--min-auc).
  2. New AUC must not regress below the current production baseline
     by more than --tolerance.

The production baseline is the metrics.json attached to the latest
GitHub Release (downloaded by the workflow). First run: no baseline,
only the floor applies.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True, help="path to new metrics.json")
    p.add_argument("--baseline", default="", help="path to baseline metrics.json (optional)")
    p.add_argument("--min-auc", type=float, default=0.95)
    p.add_argument("--tolerance", type=float, default=0.005)
    args = p.parse_args()

    new = json.loads(Path(args.new).read_text())
    new_auc = new["auc"]
    print(f"candidate model: auc={new_auc} (git={new.get('git_sha')}, data={new.get('data_hash')})")

    if new_auc < args.min_auc:
        print(f"GATE FAILED: auc {new_auc} below absolute floor {args.min_auc}")
        return 1

    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text())
        base_auc = base["auc"]
        print(f"production baseline: auc={base_auc}")
        if new_auc < base_auc - args.tolerance:
            print(
                f"GATE FAILED: auc {new_auc} regresses below baseline "
                f"{base_auc} - tolerance {args.tolerance}"
            )
            return 1
        print("GATE PASSED: candidate >= baseline (within tolerance)")
    else:
        print("no production baseline found (first run) — floor check only")
        print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

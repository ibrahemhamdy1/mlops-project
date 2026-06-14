"""Training job with MLflow experiment tracking.

Outputs (./model/):
  model.pkl            — the trained model
  metrics.json         — eval metrics + lineage (git SHA, data hash, params)
  reference_stats.json — PSI baseline for drift_check.py
  reference_data.csv   — training sample for Evidently drift reports

MLflow runs are logged to ./mlruns (file store) and uploaded as a CI
artifact. Point MLFLOW_TRACKING_URI at a real MLflow server and this
same code logs there instead — zero changes.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path("model")
PARAMS = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
PSI_BINS = 10


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def data_hash(df: pd.DataFrame) -> str:
    """Lineage: hash the exact training data (= DVC/S3 manifest hash in prod)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()[:12]


def reference_stats(x: pd.DataFrame) -> dict:
    stats = {}
    for col in x.columns:
        counts, edges = np.histogram(x[col], bins=PSI_BINS)
        stats[col] = {
            "bin_edges": edges.tolist(),
            "ratios": (counts / counts.sum()).tolist(),
        }
    return stats


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    x, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("breast-cancer-classifier")

    with mlflow.start_run():
        model = GradientBoostingClassifier(**PARAMS)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        metrics = {
            "auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, model.predict(x_test))), 4),
            "n_train": len(x_train),
            "n_test": len(x_test),
            "params": PARAMS,
            "git_sha": git_sha(),
            "data_hash": data_hash(data.frame),
        }

        # --- MLflow: the full lineage story in one run ---
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({"auc": metrics["auc"], "accuracy": metrics["accuracy"]})
        mlflow.set_tags({"git_sha": metrics["git_sha"], "data_hash": metrics["data_hash"]})
        mlflow.sklearn.log_model(
            model, "model", registered_model_name="breast-cancer-classifier"
        )

        # --- pipeline artifacts (consumed by gate / drift / Docker build) ---
        joblib.dump(model, OUT / "model.pkl")
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (OUT / "reference_stats.json").write_text(
            json.dumps(reference_stats(x_train), indent=2)
        )
        x_train.to_csv(OUT / "reference_data.csv", index=False)

        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

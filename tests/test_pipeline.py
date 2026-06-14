"""Smoke tests: data validation, a fast training run, the gate, and the API."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from validate_data import load_dataset, validate  # noqa: E402


def test_data_validation_passes():
    assert validate(load_dataset()) == []


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Run the real training script once for the test session."""
    subprocess.run(
        [sys.executable, "src/train.py"], cwd=REPO, check=True, capture_output=True
    )
    metrics = json.loads((REPO / "model" / "metrics.json").read_text())
    return metrics


def test_training_produces_artifacts(trained_model):
    assert (REPO / "model" / "model.pkl").exists()
    assert (REPO / "model" / "reference_stats.json").exists()
    assert trained_model["auc"] > 0.95


def test_gate_passes_against_itself(trained_model):
    result = subprocess.run(
        [
            sys.executable,
            "src/gate.py",
            "--new",
            "model/metrics.json",
            "--baseline",
            "model/metrics.json",
        ],
        cwd=REPO,
    )
    assert result.returncode == 0


def test_gate_fails_below_floor(tmp_path, trained_model):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"auc": 0.50}))
    result = subprocess.run(
        [sys.executable, "src/gate.py", "--new", str(bad)], cwd=REPO
    )
    assert result.returncode == 1


def test_api_predict(trained_model):
    import service

    service.load_model()
    client = TestClient(service.app)

    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200

    sample = [[14.0] * 30]
    resp = client.post("/predict", json={"instances": sample})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"][0] in (0, 1)
    assert 0.0 <= body["probabilities"][0] <= 1.0

    assert client.get("/metrics").status_code == 200


def test_api_rejects_wrong_shape(trained_model):
    import service

    client = TestClient(service.app)
    resp = client.post("/predict", json={"instances": [[1.0, 2.0]]})
    assert resp.status_code == 422

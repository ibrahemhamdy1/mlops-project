# 🚀 MLOps + DevSecOps Pipeline — GitHub Actions

A complete, working ML pipeline you can fork and run today on free GitHub runners.
It demonstrates every concept from a senior MLOps/DevSecOps interview, with real code.

```
                         ┌──────────────────────────── ci.yml (every PR) ───────────────┐
                         │ ruff · pytest · gitleaks · bandit · pip-audit · trivy config │
                         └───────────────────────────────────────────────────────────────┘

 push to main / drift event
        │
        ▼
 1. VALIDATE DATA ──► 2. TRAIN ──► 3. EVAL GATE ──► 4. BUILD IMAGE ──► 5. RELEASE ──► 6. DEPLOY
    schema, nulls,      model.pkl     candidate must    trivy scan         new prod        manual
    ranges, volume      + metrics     beat the prod     SBOM (syft)        baseline        approval,
    (fail = stop)       + lineage     baseline or       cosign sign        (model          deploy by
                        + PSI ref     pipeline FAILS    push GHCR          registry)       digest
        ▲
        │ repository_dispatch: drift-detected
 drift.yml (daily cron): PSI check of current data vs training reference
```

## MLOps tools in this pipeline

| Tool | Role here | Interview name-drop |
|---|---|---|
| **MLflow** | Experiment tracking: every run logs params, metrics, tags, model | "Tracking + Registry" |
| **pandera** | Typed data contracts in the validation gate | "data validation / GE family" |
| **Evidently** | HTML drift report in the scheduled monitor | "the standard drift tool" |
| **DVC** | Reproducible pipeline stages (`dvc repro`) | "data versioning" |
| **Trivy / Syft / cosign** | Image scan, SBOM, keyless signing | "supply chain" |
| **gitleaks / bandit / pip-audit** | Secrets, SAST, dependency audit | "shift-left" |

## What's inside

| Path | What it is |
|---|---|
| `src/validate_data.py` | Data validation gate via **pandera** typed schema + custom checks |
| `src/train.py` | Training with **MLflow tracking** (params, metrics, model, tags) + lineage in metrics.json |
| `src/gate.py` | Evaluation gate: new model must beat the production baseline |
| `src/drift_check.py` | PSI drift detection against training-time reference stats (the fast gate) |
| `src/evidently_report.py` | **Evidently** full HTML drift report (the investigation artifact) |
| `src/model_card.py` | Model card generator — governance doc attached to every release |
| `dvc.yaml` | **DVC** pipeline: `dvc repro` re-runs only stages whose inputs changed |
| `src/service.py` | FastAPI inference: `/predict`, `/healthz`, `/readyz`, Prometheus `/metrics` with prediction-distribution histograms (drift signal) |
| `tests/` | Smoke tests incl. a real fast training run + API tests |
| `Dockerfile` | Multi-stage, non-root, healthcheck, model baked in |
| `k8s/deployment.yaml` | Hardened: non-root, read-only FS, drop ALL caps, probes, preStop sleep, PDB |
| `.github/workflows/ci.yml` | DevSecOps shift-left: lint, tests, gitleaks, bandit, pip-audit, trivy config |
| `.github/workflows/train-deploy.yml` | The MLOps pipeline (6 jobs, see diagram) |
| `.github/workflows/drift.yml` | Scheduled drift monitor that triggers retraining |

## The "model registry"

GitHub Releases play the registry role: each release carries `model.pkl`,
`metrics.json` (the production baseline the gate compares against) and
`reference_stats.json` (the drift baseline). Swap for MLflow or SageMaker
Model Registry in a real platform — **the pipeline shape stays identical**.

## Setup (5 minutes)

1. **Create a repo** and push these files.
2. **Permissions**: Settings → Actions → General → Workflow permissions →
   "Read and write permissions".
3. **Manual approval gate**: Settings → Environments → create `production`
   → add yourself as required reviewer. The deploy job now waits for a human.
4. Push to `main` → watch `train-deploy` run end to end (deploy is skipped
   until step 5).
5. **(Optional) real deployment**: add repo **variable** `DEPLOY_ENABLED=true`
   and **secret** `KUBE_CONFIG` (base64 of a kubeconfig with namespace-scoped
   RBAC — see the interview guide, Q44).
6. **Demo the drift loop**: Actions → drift-monitor → Run workflow →
   check "simulate drift" → watch it dispatch a retraining run.

## Verify the supply chain (impressive in a demo)

```bash
# verify the image signature — proves it was built by THIS repo's CI
cosign verify ghcr.io/<owner>/<repo>@<digest> \
  --certificate-identity-regexp "github.com/<owner>/<repo>" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

# inspect the attested SBOM
cosign verify-attestation --type spdxjson ghcr.io/<owner>/<repo>@<digest> \
  --certificate-identity-regexp "github.com/<owner>/<repo>" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

## Run locally

```bash
pip install -r requirements-dev.txt
python src/validate_data.py
python src/train.py
python src/gate.py --new model/metrics.json
pytest -q
uvicorn src.service:app --port 8000
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"instances\": [[$(python -c 'print(",".join(["14.0"]*30))')]]}"
python src/drift_check.py --simulate-drift   # exit 2 = drift alert
python src/evidently_report.py --simulate-drift   # writes drift_report.html
python src/model_card.py                          # writes model/MODEL_CARD.md
mlflow ui --backend-store-uri file:./mlruns       # browse experiment runs
dvc init && dvc repro                             # reproducible pipeline
```

## Interview talking points this repo proves

- **Build once, promote by digest** — deploy uses the image digest, never a tag
- **Evaluation gate** — a model that doesn't beat production cannot ship
- **Lineage** — every model knows its git SHA + data hash + params
- **Human approval before prod** — GitHub environment with required reviewers
- **Supply chain** — SBOM + keyless cosign signing via GitHub OIDC (no stored keys)
- **Shift-left security** — secrets, SAST, dependency and IaC scans on every PR
- **Drift closes the loop** — scheduled PSI check dispatches retraining, but
  promotion still requires the human gate (the "trap question" answer)
- **Production-safe K8s** — non-root, read-only FS, probes, preStop, PDB

## Swap-in points for a real platform

| Demo component | Production replacement |
|---|---|
| sklearn breast-cancer dataset | your data in versioned S3 + DVC manifests |
| GitHub Releases registry | MLflow Model Registry / SageMaker Registry |
| pandera schema | Great Expectations / Glue Data Quality (same idea, bigger) |
| MLflow file store (./mlruns) | MLflow server: same code, set MLFLOW_TRACKING_URI |
| DVC without remote | `dvc remote add -d s3 s3://bucket/dvc` — data versioned in S3 |
| plain Deployment | KServe canary or Argo Rollouts |
| GitHub-hosted runners | self-hosted GPU runners / SageMaker training |

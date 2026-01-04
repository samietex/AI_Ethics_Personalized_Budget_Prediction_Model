# Responsible ML: Personalized Budget Threshold Predictor (with MLflow)

![CI](../../actions/workflows/ci.yml/badge.svg)

This project builds a **Responsible ML** workflow for predicting whether a user's budget is **above a configurable threshold** (default: **$300**). It includes:

- A reproducible training + evaluation pipeline
- **Fairness slicing** by `Education_Level`
- **Mitigation via reweighing** (baseline vs mitigated comparison)
- **Threshold sweep** to select a threshold under fairness + base-rate constraints
- **MLflow experiment tracking** (metrics, artifacts, model versions)
- A stakeholder-friendly **Streamlit demo app** (Streamlit Cloud friendly)
- Governance-ready documentation (model card, data sheet, risk assessment)

---

## Why this matters (business view)

Many personalization systems (finance, marketing, travel/entertainment, edtech) use signals similar to *budget* to tailor recommendations. If a model learns patterns correlated with socioeconomic proxies (like education), it can unintentionally create **unequal quality-of-service** between groups.

This repo demonstrates a governance-ready approach:
- measure fairness gaps,
- apply a mitigation method (reweighing),
- report trade-offs clearly (fairness vs performance),
- and keep an audit trail (MLflow artifacts + metrics).

---

## What the model predicts

**Target:**
`high_budget = 1` if `Budget (in dollars) >= threshold` else `0`

**Default threshold:** `300` (configurable)

**Features used:**
- Age
- Gender
- Education_Level *(used for fairness slicing)*
- With children?
- Recommended_Activity

---

## Responsible AI (docs + artifacts)

Governance-ready materials:

- **Model Card (template):** `docs/model_card.md`
- **Model Card (filled with latest run):** `docs/model_card.filled.md`
- **Risk Assessment:** `docs/risk_assessment.md`
- **Data Sheet:** `docs/data_sheet.md`
- **Baseline vs Mitigated summary (1-page):** `reports/artifacts/comparison.md`
- **Threshold sweep recommendation:** `reports/artifacts/threshold_sweep/<run_id>/recommended_threshold.md`

> **Tip:** The `comparison.md` and `recommended_threshold.md` files are designed to be stakeholder-friendly and easy to share.

---

## Getting started (technical)

### Option A — Conda (recommended while iterating)
```bash
conda create -n ai_ethics python=3.10 -y
conda activate ai_ethics

python -m pip install -U pip
pip install -e ".[dev]"
```

### Option B — Virtualenv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -U pip
pip install -e ".[dev]"
```

### Data setup
The training script expects a CSV with columns:

- Budget (in dollars)
- Age
- Gender
- Education_Level
- With children?
- Recommended_Activity

Recommended local layout:

```
data/
  udacity_ai_ethics_project_data.csv
```

**Note:** `data/` is typically gitignored to avoid committing private datasets.

---

## Train (baseline + mitigation) and log to MLflow

```bash
python -m budget_fairness.train \
  --data-path data/udacity_ai_ethics_project_data.csv \
  --model logreg \
  --sample-n 50000
```

View MLflow UI (audit trail):

```bash
mlflow ui --backend-store-uri ./mlruns
```

### What gets logged to MLflow

- **Params:** threshold, test_size, random_state, mitigation method, etc.
- **Metrics:** baseline + mitigated performance and fairness
- **Artifacts:** confusion matrices, classification reports, fairness reports (JSON/MD), comparison summary
- **Models:** baseline and mitigated pipelines

---

## Demo app (Streamlit)

A lightweight Streamlit app is included for stakeholder-friendly demos. It supports two modes:

- **Streamlit Cloud / hosted:** loads bundled demo models from `models/` (no MLflow required)
- **Local developer workflow:** can load models from the latest MLflow run (`./mlruns`) when available

### Install app dependencies
```bash
pip install -e ".[app]"
```

### Run the app locally
```bash
streamlit run app/ai_ethics_streamlit.py
```

### Bundled demo models (recommended for Streamlit Cloud)

To make the app work on Streamlit Cloud (no local `mlruns/`), include demo model artifacts in the repo:

```
models/
  demo_baseline/
  demo_mitigated/
```

The app will automatically prefer these bundled models when present.

If you deploy to Streamlit Cloud, use Python 3.10–3.12 (recommended) and pin your dependencies.

---

## Fairness evaluation (what we measure)

We compute group fairness using `Education_Level`:

- **Privileged group:** Bachelor's / Master's (configurable)
- **Unprivileged group:** all others

Metrics logged include:

- **SPD** (statistical parity difference)
- **DI** (disparate impact)
- **EOD** (equal opportunity difference)
- **AOD** (average odds difference)
- **PPV diff** (predictive parity difference)

Each run logs:

- `baseline_fairness_*`
- `mitigated_fairness_*`
- plus deltas (`delta_*`) for quick comparison.

---

## Mitigation method (what we do)

We apply **reweighing** during training:

1. compute instance weights based on group membership and label frequency
2. fit Logistic Regression using `sample_weight`
3. compare baseline vs mitigated results in the same MLflow run

Summary artifact:

```
reports/artifacts/comparison.md
```

---

## Threshold sweep (fairness vs threshold)

Budget threshold selection can strongly affect both performance and fairness. To avoid choosing a threshold arbitrarily, we run a threshold sweep and evaluate:

- Performance metrics (accuracy, precision/recall, F1, ROC AUC)
- Fairness metrics (especially DI) under the same `Education_Level` group definition
- A base-rate constraint to avoid trivial thresholds where almost everyone is positive/negative

### Constraints used

- **Fairness constraint:** mitigated DI ∈ [0.8, 1.25]
- **Base-rate constraint:** positive rate P(y=1) ∈ [0.3, 0.7]

### Run the sweep

```bash
python -m budget_fairness.threshold_sweep \
  --data-path data/udacity_ai_ethics_project_data.csv \
  --min-threshold 100 \
  --max-threshold 800 \
  --step 50 \
  --di-min 0.8 \
  --di-max 1.25 \
  --pos-min 0.3 \
  --pos-max 0.7 \
  --sample-n 50000
```

### Sweep artifacts (saved locally + logged to MLflow)

Artifacts are written to:

```
reports/artifacts/threshold_sweep/<run_id>/
```

Key files:

- `threshold_sweep_results.csv` / `.json` (all thresholds + metrics)
- `fairness_vs_threshold_di.png` (DI vs threshold plot)
- `recommended_threshold.md` (recommended threshold and trade-off summary)

---

## Generate a filled model card from the latest training run

After training, generate an updated model card from `reports/artifacts/last_run.json`:

```bash
python -m budget_fairness.reporting --run-json reports/artifacts/last_run.json
```

**Output:**

```
docs/model_card.filled.md
```

---

## Project structure

```
app/                       Streamlit demo app
docs/                      Responsible AI documentation (model card, risk, data sheet)
models/                    Bundled demo models (for Streamlit Cloud)
mlruns/                    Local MLflow store (ignored)
reports/artifacts/         Generated artifacts (baseline/mitigated + sweep outputs)
src/budget_fairness/       Core package (data, training, fairness, mitigation, reporting, sweeps)
tests/                     Unit tests
```

---

## CI/CD

This repo uses GitHub Actions for continuous integration:

- Linting with `ruff`
- Tests with `pytest`

Workflow file:

```
.github/workflows/ci.yml
```

### Requirements file (optional)

If you prefer a `requirements.txt` workflow (e.g., for certain deployment targets), you can generate one from your packaging config and keep it in sync. For this repo, the canonical source of dependencies is the project configuration (editable install):

```bash
pip install -e ".[dev]"
```

---

## What's next (planned improvements)

- Coverage reporting (with a minimum threshold)
- Dependency caching in CI for faster runs
- Dockerfile for fully reproducible local runs
- Model selection rules (auto-pick baseline vs mitigated under constraints)
- Scheduled evaluation runs (GitHub Actions cron) for monitoring-style reporting

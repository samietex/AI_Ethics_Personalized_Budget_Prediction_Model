from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Budget Threshold Predictor", layout="centered")

st.title("Personalized Budget Threshold Predictor")
st.caption("Baseline vs Mitigated (reweighing) — Responsible ML demo")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_demo_model(model_dir: Path):
    return mlflow.pyfunc.load_model(str(model_dir))


@st.cache_resource(show_spinner=False)
def load_mlflow_run_model(run_id: str, artifact_subpath: str):
    uri = f"runs:/{run_id}/{artifact_subpath}"
    return mlflow.pyfunc.load_model(uri)


def get_model(model_choice: str, run_id: str | None):
    """
    Priority:
    1) Repo-bundled demo models (Streamlit Cloud friendly)
    2) Local MLflow run models (developer workflow)
    """
    demo_baseline = Path("models/demo_baseline")
    demo_mitigated = Path("models/demo_mitigated")

    if model_choice.startswith("Mitigated") and demo_mitigated.exists():
        return load_demo_model(demo_mitigated), "demo", "models/demo_mitigated"
    if model_choice.startswith("Baseline") and demo_baseline.exists():
        return load_demo_model(demo_baseline), "demo", "models/demo_baseline"

    if run_id is None:
        raise RuntimeError("No demo models found and no MLflow run_id available.")

    artifact_path = "mitigated_model" if model_choice.startswith("Mitigated") else "baseline_model"
    return load_mlflow_run_model(run_id, artifact_path), "mlflow", f"runs:/{run_id}/{artifact_path}"


def load_last_run() -> dict[str, Any] | None:
    p = Path("reports/artifacts/last_run.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_latest_run_id_from_mlruns() -> str | None:
    """
    Best-effort fallback: find most recently modified run in local ./mlruns store.
    Structure: mlruns/<exp_id>/<run_id>/meta.yaml
    """
    mlruns = Path("mlruns")
    if not mlruns.exists():
        return None

    run_infos: list[tuple[float, str]] = []
    for meta in mlruns.glob("*/*/meta.yaml"):
        try:
            ts = meta.stat().st_mtime
            run_id = meta.parent.name
            run_infos.append((ts, run_id))
        except Exception:
            continue

    if not run_infos:
        return None

    run_infos.sort(reverse=True)
    return run_infos[0][1]


def safe_predict_proba(model, X: pd.DataFrame) -> float | None:
    """
    Try to extract probability for class 1.
    Note: MLflow pyfunc often only exposes .predict, so this may return None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return float(proba[0][1])
    return None


# ----------------------------
# Load run metadata (optional)
# ----------------------------
last_run = load_last_run()
run_id: str | None = None
train_threshold: float | None = None

if last_run and "run_id" in last_run:
    run_id = str(last_run["run_id"])

if run_id is None:
    run_id = find_latest_run_id_from_mlruns()

if last_run:
    train_threshold = last_run.get("threshold")


# ----------------------------
# Sidebar: inputs
# ----------------------------
with st.sidebar:
    st.header("Inputs")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    education = st.selectbox(
        "Education_Level",
        [
            "High School Grad",
            "Did Not Graduate HS",
            "Bachelor Degree",
            "Master Degree",
            "Other",
        ],
    )

    with_children = st.selectbox("With children?", [0, 1])

    activity = st.selectbox("Recommended_Activity", ["A", "B", "C", "D", "E"])

    model_choice = st.radio(
        "Model",
        options=["Mitigated (reweighing)", "Baseline"],
        index=0,
    )

    st.divider()
    st.caption(
        "Uses bundled demo models if present; otherwise loads from the latest local MLflow run."
    )


# ----------------------------
# Build input frame
# ----------------------------
X = pd.DataFrame(
    [
        {
            "Age": age,
            "Gender": gender,
            "Education_Level": education,
            "With children?": with_children,
            "Recommended_Activity": activity,
        }
    ]
)

st.subheader("Input")
st.dataframe(X, use_container_width=True)


# ----------------------------
# Predict
# ----------------------------
st.subheader("Prediction")

# If we have neither demo models nor local mlruns, we can’t proceed.
if not Path("models/demo_baseline").exists() and not Path("models/demo_mitigated").exists():
    if run_id is None:
        st.error(
            "No demo models found and no MLflow run found.\n\n"
            "1) Train locally:\n"
            "`python -m budget_fairness.train --data-path data/udacity_ai_ethics_project_data.csv --sample-n 50000`\n\n"
            "2) Export demo models for Streamlit Cloud:\n"
            "`python scripts/export_demo_models.py`\n"
        )
        st.stop()

with st.expander("Debug info", expanded=False):
    st.write(f"Detected run_id: `{run_id}`")
    st.write(f"Demo baseline exists: `{Path('models/demo_baseline').exists()}`")
    st.write(f"Demo mitigated exists: `{Path('models/demo_mitigated').exists()}`")
    st.write(f"Local mlruns exists: `{Path('mlruns').exists()}`")

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    with st.spinner("Loading model + predicting..."):
        model, source, source_ref = get_model(model_choice, run_id)

        yhat = model.predict(X)
        pred = int(yhat[0])

        proba1 = safe_predict_proba(model, X)

    label = "Above threshold ✅" if pred == 1 else "Below threshold ❌"
    st.success(f"Prediction: **{pred}** → **{label}**")

    st.caption(f"Model source: **{source}** (`{source_ref}`)")

    if proba1 is not None:
        st.metric("P(Above threshold)", f"{proba1:.3f}")

    if train_threshold is not None:
        st.caption(f"Training threshold: {train_threshold}")


# ----------------------------
# Responsible AI links
# ----------------------------
st.subheader("Responsible AI documentation")

st.markdown(
    """
- Model card (template): `docs/model_card.md`
- Model card (filled): `docs/model_card.filled.md`
- Risk assessment: `docs/risk_assessment.md`
- Data sheet: `docs/data_sheet.md`
- Baseline vs mitigated summary: `reports/artifacts/comparison.md`
- Threshold sweep recommendation: `reports/artifacts/threshold_sweep/.../recommended_threshold.md`
"""
)

if last_run:
    st.subheader("Latest run summary (last_run.json)")
    st.json(last_run)

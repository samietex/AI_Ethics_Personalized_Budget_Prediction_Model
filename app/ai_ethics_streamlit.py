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
def load_last_run() -> dict[str, Any] | None:
    p = Path("reports/artifacts/last_run.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_latest_run_id_from_mlruns() -> str | None:
    """
    Fallback: try to find the latest MLflow run_id in local ./mlruns.
    This is best-effort (works for local file store).
    """
    mlruns = Path("mlruns")
    if not mlruns.exists():
        return None

    run_infos: list[tuple[float, str]] = []
    # structure: mlruns/<exp_id>/<run_id>/meta.yaml
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


@st.cache_resource(show_spinner=False)
def load_model(run_id: str, artifact_subpath: str):
    """
    Load an MLflow logged model from a given run id and artifact path.
    Example artifact_subpath: "baseline_model" or "mitigated_model"
    """
    uri = f"runs:/{run_id}/{artifact_subpath}"
    return mlflow.pyfunc.load_model(uri)


def safe_predict_proba(model, X: pd.DataFrame) -> float | None:
    """
    Try to extract probability for class 1 from either sklearn pipeline or mlflow pyfunc.
    """
    # MLflow pyfunc wrappers typically expose predict only.
    # But sklearn models loaded directly may have predict_proba.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return float(proba[0][1])
    return None


# ----------------------------
# Load run metadata
# ----------------------------
last_run = load_last_run()
run_id = None
train_threshold = None

if last_run and "run_id" in last_run:
    run_id = last_run["run_id"]

# fallback if last_run.json missing
if run_id is None:
    run_id = find_latest_run_id_from_mlruns()

if last_run:
    # optional: threshold might live in params logged to MLflow instead;
    # but if you store it in last_run in future, we’ll use it.
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

    activity = st.selectbox(
        "Recommended_Activity",
        ["A", "B", "C", "D", "E"],
    )

    model_choice = st.radio(
        "Model",
        options=["Mitigated (reweighing)", "Baseline"],
        index=0,
    )

    st.divider()
    st.caption("Prediction is based on the latest local MLflow run.")


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
# Predict button
# ----------------------------
st.subheader("Prediction")

if run_id is None:
    st.error(
        "No MLflow run found. Train a model first:\n\n"
        "`python -m budget_fairness.train --data-path data/udacity_ai_ethics_project_data.csv --sample-n 50000`"
    )
    st.stop()

artifact_path = "mitigated_model" if model_choice.startswith("Mitigated") else "baseline_model"

with st.expander("Debug info", expanded=False):
    st.write(f"Run ID: `{run_id}`")
    st.write(f"Artifact path: `{artifact_path}`")
    if Path("mlruns").exists():
        st.write("Using local MLflow store: `./mlruns`")

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    with st.spinner("Loading model + predicting..."):
        model = load_model(run_id, artifact_path)

        yhat = model.predict(X)
        # yhat is typically an array-like
        pred = int(yhat[0])

        proba1 = safe_predict_proba(model, X)

    label = "Above threshold ✅" if pred == 1 else "Below threshold ❌"
    st.success(f"Prediction: **{pred}** → **{label}**")

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

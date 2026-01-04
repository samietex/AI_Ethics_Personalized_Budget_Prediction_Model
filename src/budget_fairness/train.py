from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import BudgetConfig
from .data import load_raw_csv, to_model_frame
from .evaluation import evaluate_classifier
from .fairness import compute_fairness_by_education
from .mitigation import compute_reweighing_weights

def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    # Identify columns by dtype (robust to column order changes)
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def build_logreg_model(X: pd.DataFrame, *, random_state: int) -> Pipeline:
    preprocess = build_preprocess(X)
    clf = LogisticRegression(max_iter=300, solver="liblinear", random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocess), ("model", clf)])


def _plot_confusion_matrix(y_true, y_pred, outpath: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format="d")
    disp.ax_.set_title("Confusion matrix")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()

def _fairness_markdown(fairness) -> str:
    priv = fairness.group_rates["privileged"]
    unpriv = fairness.group_rates["unprivileged"]
    fm = fairness.fairness_metrics

    return f"""# Fairness Report (Education_Level)

## Group definition
- Privileged: {fairness.privileged_label}
- Unprivileged: {fairness.unprivileged_label}

## Group counts
- Privileged: {fairness.group_counts["privileged"]}
- Unprivileged: {fairness.group_counts["unprivileged"]}

## Group rates (positive prediction: y_hat = 1)
| Metric | Privileged | Unprivileged |
|---|---:|---:|
| Selection rate P(y_hat=1) | {priv["selection_rate"]:.4f} | {unpriv["selection_rate"]:.4f} |
| TPR P(y_hat=1 | y=1) | {priv["tpr"]:.4f} | {unpriv["tpr"]:.4f} |
| FPR P(y_hat=1 | y=0) | {priv["fpr"]:.4f} | {unpriv["fpr"]:.4f} |
| PPV P(y=1 | y_hat=1) | {priv["ppv"]:.4f} | {unpriv["ppv"]:.4f} |

## Fairness metrics
- Statistical Parity Difference (SPD) = SR_unpriv - SR_priv = **{fm["statistical_parity_difference"]:.4f}**
- Disparate Impact (DI) = SR_unpriv / SR_priv = **{fm["disparate_impact"]:.4f}**
- Equal Opportunity Difference (EOD) = TPR_unpriv - TPR_priv = **{fm["equal_opportunity_difference"]:.4f}**
- Average Odds Difference (AOD) = 0.5[(FPR_unpriv-FPR_priv)+(TPR_unpriv-TPR_priv)] = **{fm["average_odds_difference"]:.4f}**
- Predictive Parity Difference (PPV diff) = PPV_unpriv - PPV_priv = **{fm["predictive_parity_difference"]:.4f}**
"""

def _comparison_markdown(base_eval, base_fair, mit_eval, mit_fair) -> str:
    be, bf = base_eval.metrics, base_fair.fairness_metrics
    me, mf = mit_eval.metrics, mit_fair.fairness_metrics

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    def delta(new, old):
        try:
            return f"{float(new) - float(old):+.4f}"
        except Exception:
            return "n/a"

    lines = []
    lines.append("# Baseline vs Mitigated (Reweighing) — Summary\n")

    lines.append("## Performance (higher is better)\n")
    lines.append("| Metric | Baseline | Mitigated | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in be or k in me:
            b = be.get(k, float("nan"))
            m = me.get(k, float("nan"))
            lines.append(f"| {k} | {fmt(b)} | {fmt(m)} | {delta(m, b)} |")
    lines.append("")

    lines.append("## Fairness (Education_Level slicing)\n")
    lines.append("| Metric | Baseline | Mitigated | Δ |")
    lines.append("|---|---:|---:|---:|")
    mapping = [
        ("statistical_parity_difference", "SPD (SR_unpriv - SR_priv)"),
        ("disparate_impact", "DI (SR_unpriv / SR_priv)"),
        ("equal_opportunity_difference", "EOD (TPR_unpriv - TPR_priv)"),
        ("average_odds_difference", "AOD"),
        ("predictive_parity_difference", "PPV diff (PPV_unpriv - PPV_priv)"),
    ]
    for key, label in mapping:
        b = bf.get(key, float("nan"))
        m = mf.get(key, float("nan"))
        lines.append(f"| {label} | {fmt(b)} | {fmt(m)} | {delta(m, b)} |")
    lines.append("")

    # Simple narrative trade-off line (you can adjust wording later)
    lines.append("## Trade-off (plain English)\n")
    lines.append(
        f"- **Disparate Impact (DI)** changed from **{fmt(bf.get('disparate_impact'))}** to **{fmt(mf.get('disparate_impact'))}** "
        f"({delta(mf.get('disparate_impact'), bf.get('disparate_impact'))}).\n"
        f"- **F1 score** changed from **{fmt(be.get('f1'))}** to **{fmt(me.get('f1'))}** "
        f"({delta(me.get('f1'), be.get('f1'))}).\n"
        "\n"
        "Interpretation: reweighing aims to reduce group disparity at training time. "
        "If fairness improves while performance drops slightly, that’s a common and expected trade-off. "
        "The best choice depends on business constraints, legal/ethical requirements, and risk tolerance.\n"
    )

    return "\n".join(lines)




def run_training(
    *,
    data_path: str,
    model_type: str = "logreg",
    config: BudgetConfig = BudgetConfig(),
) -> dict:
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    mlflow.set_experiment(config.experiment_name)

    raw = load_raw_csv(data_path)

    # Optional downsampling for faster local iteration
    if config.sample_n is not None:
        raw = raw.sample(n=int(config.sample_n), random_state=config.random_state)

    X, y = to_model_frame(raw, threshold=config.threshold)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    if model_type != "logreg":
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Build two identical pipelines (baseline + mitigated)
    baseline_model = build_logreg_model(X_train, random_state=config.random_state)
    mitigated_model = build_logreg_model(X_train, random_state=config.random_state)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": model_type,
                "threshold": config.threshold,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "sample_n": config.sample_n if config.sample_n is not None else -1,
                "mitigation": "reweighing",
            }
        )

        artifacts_dir = Path("reports") / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)


        # -----------------------
        # 1) BASELINE TRAIN/EVAL
        # -----------------------
        baseline_model.fit(X_train, y_train)

        y_pred_base = baseline_model.predict(X_test)
        y_prob_base = None
        if hasattr(baseline_model, "predict_proba"):
            y_prob_base = baseline_model.predict_proba(X_test)[:, 1]

        base_eval = evaluate_classifier(y_test, y_pred_base, y_prob=y_prob_base)
        base_fair = compute_fairness_by_education(
            X_test,
            y_test,
            y_pred_base,
            privileged_education_levels=tuple(config.privileged_education_levels),
            education_col="Education_Level",
        )

        mlflow.log_metrics(_prefix_metrics("baseline_perf", base_eval.metrics))
        mlflow.log_metrics(
            {
                "baseline_fairness_spd": base_fair.fairness_metrics["statistical_parity_difference"],
                "baseline_fairness_di": base_fair.fairness_metrics["disparate_impact"],
                "baseline_fairness_eod": base_fair.fairness_metrics["equal_opportunity_difference"],
                "baseline_fairness_aod": base_fair.fairness_metrics["average_odds_difference"],
                "baseline_fairness_ppv_diff": base_fair.fairness_metrics["predictive_parity_difference"],
            }
        )

        # Baseline artifacts
        base_dir = artifacts_dir / "baseline"
        base_dir.mkdir(parents=True, exist_ok=True)

        (base_dir / "fairness_metrics.json").write_text(
            json.dumps(base_fair.to_json(), indent=2), encoding="utf-8"
        )
        (base_dir / "fairness_report.md").write_text(_fairness_markdown(base_fair), encoding="utf-8")

        cm_path = base_dir / "confusion_matrix.png"
        _plot_confusion_matrix(y_test, y_pred_base, cm_path)
        (base_dir / "classification_report.txt").write_text(base_eval.report, encoding="utf-8")

        mlflow.log_artifact(str(base_dir / "fairness_metrics.json"))
        mlflow.log_artifact(str(base_dir / "fairness_report.md"))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(base_dir / "classification_report.txt"))

        # --------------------------
        # 2) MITIGATED TRAIN/EVAL
        # --------------------------
        weights = compute_reweighing_weights(
            X_train,
            y_train,
            privileged_education_levels=tuple(config.privileged_education_levels),
            education_col="Education_Level",
        )

        # Pipeline passes sample_weight to final estimator using step name: model__
        mitigated_model.fit(X_train, y_train, model__sample_weight=weights)

        y_pred_mit = mitigated_model.predict(X_test)
        y_prob_mit = None
        if hasattr(mitigated_model, "predict_proba"):
            y_prob_mit = mitigated_model.predict_proba(X_test)[:, 1]

        mit_eval = evaluate_classifier(y_test, y_pred_mit, y_prob=y_prob_mit)
        mit_fair = compute_fairness_by_education(
            X_test,
            y_test,
            y_pred_mit,
            privileged_education_levels=tuple(config.privileged_education_levels),
            education_col="Education_Level",
        )

        mlflow.log_metrics(_prefix_metrics("mitigated_perf", mit_eval.metrics))
        mlflow.log_metrics(
            {
                "mitigated_fairness_spd": mit_fair.fairness_metrics["statistical_parity_difference"],
                "mitigated_fairness_di": mit_fair.fairness_metrics["disparate_impact"],
                "mitigated_fairness_eod": mit_fair.fairness_metrics["equal_opportunity_difference"],
                "mitigated_fairness_aod": mit_fair.fairness_metrics["average_odds_difference"],
                "mitigated_fairness_ppv_diff": mit_fair.fairness_metrics["predictive_parity_difference"],
            }
        )

        # Mitigated artifacts
        mit_dir = artifacts_dir / "mitigated_reweighing"
        mit_dir.mkdir(parents=True, exist_ok=True)

        (mit_dir / "fairness_metrics.json").write_text(
            json.dumps(mit_fair.to_json(), indent=2), encoding="utf-8"
        )
        (mit_dir / "fairness_report.md").write_text(_fairness_markdown(mit_fair), encoding="utf-8")

        cm_path2 = mit_dir / "confusion_matrix.png"
        _plot_confusion_matrix(y_test, y_pred_mit, cm_path2)
        (mit_dir / "classification_report.txt").write_text(mit_eval.report, encoding="utf-8")

        mlflow.log_artifact(str(mit_dir / "fairness_metrics.json"))
        mlflow.log_artifact(str(mit_dir / "fairness_report.md"))
        mlflow.log_artifact(str(cm_path2))
        mlflow.log_artifact(str(mit_dir / "classification_report.txt"))

        # Comparison artifact
        comparison_path = artifacts_dir / "comparison.md"
        comparison_path.write_text(
            _comparison_markdown(base_eval, base_fair, mit_eval, mit_fair),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(comparison_path))

        # --------------------------
        # 3) Log deltas (optional)
        # --------------------------
        mlflow.log_metrics(
            {
                "delta_fairness_spd": mit_fair.fairness_metrics["statistical_parity_difference"]
                - base_fair.fairness_metrics["statistical_parity_difference"],
                "delta_fairness_di": mit_fair.fairness_metrics["disparate_impact"]
                - base_fair.fairness_metrics["disparate_impact"],
                "delta_fairness_eod": mit_fair.fairness_metrics["equal_opportunity_difference"]
                - base_fair.fairness_metrics["equal_opportunity_difference"],
                "delta_perf_accuracy": mit_eval.metrics["accuracy"] - base_eval.metrics["accuracy"],
                "delta_perf_f1": mit_eval.metrics["f1"] - base_eval.metrics["f1"],
            }
        )

        # Log ONE model artifact (optional choice)
        # Keep baseline as the logged model for now; later we can choose best-by-metric.
        mlflow.sklearn.log_model(baseline_model, artifact_path="baseline_model")
        mlflow.sklearn.log_model(mitigated_model, artifact_path="mitigated_model")

        out = {
            "run_id": run.info.run_id,
            "baseline": {"metrics": base_eval.metrics, "fairness": base_fair.fairness_metrics},
            "mitigated": {"metrics": mit_eval.metrics, "fairness": mit_fair.fairness_metrics},
            "artifact_dir": str(artifacts_dir),
        }

        (artifacts_dir / "last_run.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifacts_dir / "last_run.json"))

        return out



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the budget threshold classifier and log to MLflow.")
    p.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    p.add_argument("--model", default="logreg", choices=["logreg"], help="Model type to train.")
    p.add_argument("--threshold", type=float, default=BudgetConfig.threshold, help="Budget threshold.")
    p.add_argument("--test-size", type=float, default=BudgetConfig.test_size, help="Test size fraction.")
    p.add_argument("--random-state", type=int, default=BudgetConfig.random_state, help="Random seed.")
    p.add_argument("--sample-n", type=int, default=None, help="Optional row sample size for faster runs.")
    p.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI.")
    p.add_argument("--experiment-name", default=BudgetConfig.experiment_name, help="MLflow experiment name.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = BudgetConfig(
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_n=args.sample_n,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    out = run_training(data_path=args.data_path, model_type=args.model, config=cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

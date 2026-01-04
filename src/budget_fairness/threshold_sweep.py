from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import lines
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BudgetConfig
from .data import load_raw_csv, to_model_frame
from .evaluation import evaluate_classifier
from .fairness import compute_fairness_by_education
from .mitigation import compute_reweighing_weights
from .train import build_logreg_model  # reuse the same pipeline builder


def _parse_thresholds(min_t: float, max_t: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_t < min_t:
        raise ValueError("max-threshold must be >= min-threshold")
    vals = np.arange(min_t, max_t + 1e-9, step, dtype=float)
    return [float(v) for v in vals]


def _plot_di_vs_threshold(df: pd.DataFrame, outpath: Path, di_min: float, di_max: float) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = df["threshold"].to_numpy()
    y_base = df["baseline_di"].to_numpy()
    y_mit = df["mitigated_di"].to_numpy()

    plt.figure()
    plt.plot(x, y_base, marker="o", label="Baseline DI")
    plt.plot(x, y_mit, marker="o", label="Mitigated DI (reweighing)")
    plt.axhline(di_min, linestyle="--", label=f"DI lower bound ({di_min})")
    plt.axhline(di_max, linestyle="--", label=f"DI upper bound ({di_max})")
    plt.title("Disparate Impact (DI) vs Threshold")
    plt.xlabel("Budget threshold ($)")
    plt.ylabel("Disparate Impact (SR_unpriv / SR_priv)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _recommend_threshold(
    df: pd.DataFrame,
    *,
    di_min: float,
    di_max: float,
    pos_min: float,
    pos_max: float,
    optimize_metric: str = "mitigated_f1",
) -> dict:
    """
    Recommendation policy:
      1) Prefer thresholds where mitigated DI is within [di_min, di_max]
      2) Among those, choose threshold maximizing `optimize_metric` (default mitigated_f1)
      3) If none satisfy DI constraint, choose threshold with DI closest to the interval
         (min distance to [di_min, di_max]), then highest `optimize_metric`.
    """
    work = df.copy()

    if optimize_metric not in work.columns:
        raise ValueError(f"optimize_metric '{optimize_metric}' not found in columns")

    # Only consider rows with finite DI and metric
    work = work.replace([np.inf, -np.inf], np.nan)

    # Candidates within DI bounds
    in_bounds = work[
        (work["mitigated_di"].notna())
        & (work["mitigated_di"] >= di_min)
        & (work["mitigated_di"] <= di_max)
        & (work[optimize_metric].notna())
    ]

    # Base-rate constraint: avoid trivial thresholds where almost everyone is positive/negative
    in_bounds = in_bounds[
        (in_bounds["positive_rate"].notna())
        & (in_bounds["positive_rate"] >= pos_min)
        & (in_bounds["positive_rate"] <= pos_max)
    ]

    if len(in_bounds) > 0:
        best = in_bounds.sort_values(by=optimize_metric, ascending=False).iloc[0]
        reason = "within_di_bounds_maximize_metric"
    else:
        # Distance to interval
        def dist_to_interval(di: float) -> float:
            if np.isnan(di):
                return np.inf
            if di < di_min:
                return di_min - di
            if di > di_max:
                return di - di_max
            return 0.0

        work["di_distance"] = work["mitigated_di"].apply(dist_to_interval)
        work2 = work[
            (work["di_distance"].notna())
            & (work[optimize_metric].notna())
            & (work["positive_rate"].notna())
            & (work["positive_rate"] >= pos_min)
            & (work["positive_rate"] <= pos_max)
        ]
        best = work2.sort_values(by=["di_distance", optimize_metric], ascending=[True, False]).iloc[
            0
        ]
        reason = "no_threshold_meets_di_bounds_closest_di_then_maximize_metric"

    return {
        "reason": reason,
        "recommended_threshold": float(best["threshold"]),
        "recommended_metric_name": optimize_metric,
        "recommended_metric_value": float(best[optimize_metric]),
        "recommended_mitigated_di": float(best["mitigated_di"])
        if not pd.isna(best["mitigated_di"])
        else None,
        "recommended_baseline_f1": float(best["baseline_f1"]) if "baseline_f1" in best else None,
        "recommended_mitigated_f1": float(best["mitigated_f1"]) if "mitigated_f1" in best else None,
    }


def _summary_markdown(df: pd.DataFrame, rec: dict, di_min: float, di_max: float) -> str:
    thr = rec["recommended_threshold"]
    row = df[df["threshold"] == thr].iloc[0]

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    meets = (
        pd.notna(row["mitigated_di"])
        and (row["mitigated_di"] >= di_min)
        and (row["mitigated_di"] <= di_max)
    )

    top = df.copy()
    top = top.replace([np.inf, -np.inf], np.nan)
    top = top.sort_values(by="mitigated_f1", ascending=False).head(8)

    lines = []
    lines.append("# Threshold Sweep — Recommendation\n")
    lines.append(f"**DI constraint:** mitigated DI ∈ [{di_min}, {di_max}]")
    lines.append(f"**Recommended threshold:** **${thr:.2f}**")
    lines.append(f"**Meets DI constraint?** {'✅ Yes' if meets else '⚠️ No (closest available)'}")
    lines.append(f"**Selection rule:** {rec['reason']}\n")

    lines.append("## At recommended threshold\n")
    lines.append("| Metric | Baseline | Mitigated (reweighing) | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        b = row.get(f"baseline_{k}", np.nan)
        m = row.get(f"mitigated_{k}", np.nan)
        delta_str = fmt(m - b) if pd.notna(m) and pd.notna(b) else "n/a"
        lines.append(f"| {k} | {fmt(b)} | {fmt(m)} | {delta_str} |")

    lines.append("")
    lines.append(f"- **Positive rate P(y=1)** at threshold: **{fmt(row['positive_rate'])}**\n")

    lines.append("## Fairness (DI/SPD/EOD)\n")
    lines.append("| Metric | Baseline | Mitigated | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in ["spd", "di", "eod", "aod", "ppv_diff"]:
        b = row.get(f"baseline_{k}", np.nan)
        m = row.get(f"mitigated_{k}", np.nan)
        delta_str = fmt(m - b) if pd.notna(m) and pd.notna(b) else "n/a"
        lines.append(f"| {k} | {fmt(b)} | {fmt(m)} | {delta_str} |")

    lines.append("")

    lines.append("## Top thresholds by mitigated F1 (for reference)\n")
    lines.append("| threshold | mitigated_f1 | mitigated_di | baseline_f1 | baseline_di |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        lines.append(
            f"| {r['threshold']:.2f} | {fmt(r['mitigated_f1'])} | {fmt(r['mitigated_di'])} | {fmt(r['baseline_f1'])} | {fmt(r['baseline_di'])} |"
        )

    lines.append(
        "| "
        f"{r['threshold']:.2f} | {fmt(r['positive_rate'])} | {fmt(r['mitigated_f1'])} | "
        f"{fmt(r['mitigated_di'])} | {fmt(r['baseline_f1'])} | {fmt(r['baseline_di'])} |"
    )

    lines.append("|---:|---:|---:|---:|---:|---:|")
    ...
    lines.append(
        f"| {r['threshold']:.2f} | {fmt(r['positive_rate'])} | {fmt(r['mitigated_f1'])} | {fmt(r['mitigated_di'])} | {fmt(r['baseline_f1'])} | {fmt(r['baseline_di'])} |"
    )

    lines.append("\nArtifacts:")
    lines.append("- `threshold_sweep_results.csv` / `.json`")
    lines.append("- `fairness_vs_threshold_di.png`")
    return "\n".join(lines)


def run_threshold_sweep(
    *,
    data_path: str,
    thresholds: list[float],
    di_min: float,
    di_max: float,
    pos_min: float,
    pos_max: float,
    config: BudgetConfig,
) -> dict:
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    mlflow.set_experiment(config.experiment_name)

    raw = load_raw_csv(data_path)
    if config.sample_n is not None:
        raw = raw.sample(n=int(config.sample_n), random_state=config.random_state)

    rows: list[dict] = []

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "sweep": True,
                "model_type": "logreg",
                "mitigation": "reweighing",
                "threshold_min": min(thresholds),
                "threshold_max": max(thresholds),
                "threshold_step": thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0,
                "di_min": di_min,
                "di_max": di_max,
                "pos_min": pos_min,
                "pos_max": pos_max,
                "test_size": config.test_size,
                "random_state": config.random_state,
                "sample_n": config.sample_n if config.sample_n is not None else -1,
            }
        )

        run_id = run.info.run_id
        out_dir = Path("reports") / "artifacts" / "threshold_sweep" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for thr in thresholds:
            # rebuild label for each threshold
            X, y = to_model_frame(raw, threshold=thr)

            pos_rate = float(np.mean(y == 1))

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=y,
            )

            # Baseline
            base_model = build_logreg_model(X_train, random_state=config.random_state)
            base_model.fit(X_train, y_train)

            y_pred_base = base_model.predict(X_test)
            y_prob_base = (
                base_model.predict_proba(X_test)[:, 1]
                if hasattr(base_model, "predict_proba")
                else None
            )

            base_eval = evaluate_classifier(y_test, y_pred_base, y_prob=y_prob_base)
            base_fair = compute_fairness_by_education(
                X_test,
                y_test,
                y_pred_base,
                privileged_education_levels=tuple(config.privileged_education_levels),
                education_col="Education_Level",
            )

            # Mitigated (reweighing)
            weights = compute_reweighing_weights(
                X_train,
                y_train,
                privileged_education_levels=tuple(config.privileged_education_levels),
                education_col="Education_Level",
            )
            mit_model = build_logreg_model(X_train, random_state=config.random_state)
            mit_model.fit(X_train, y_train, model__sample_weight=weights)

            y_pred_mit = mit_model.predict(X_test)
            y_prob_mit = (
                mit_model.predict_proba(X_test)[:, 1]
                if hasattr(mit_model, "predict_proba")
                else None
            )

            mit_eval = evaluate_classifier(y_test, y_pred_mit, y_prob=y_prob_mit)
            mit_fair = compute_fairness_by_education(
                X_test,
                y_test,
                y_pred_mit,
                privileged_education_levels=tuple(config.privileged_education_levels),
                education_col="Education_Level",
            )

            row = {
                "threshold": float(thr),
                # perf
                "baseline_accuracy": base_eval.metrics.get("accuracy", np.nan),
                "baseline_precision": base_eval.metrics.get("precision", np.nan),
                "baseline_recall": base_eval.metrics.get("recall", np.nan),
                "baseline_f1": base_eval.metrics.get("f1", np.nan),
                "baseline_roc_auc": base_eval.metrics.get("roc_auc", np.nan),
                "mitigated_accuracy": mit_eval.metrics.get("accuracy", np.nan),
                "mitigated_precision": mit_eval.metrics.get("precision", np.nan),
                "mitigated_recall": mit_eval.metrics.get("recall", np.nan),
                "mitigated_f1": mit_eval.metrics.get("f1", np.nan),
                "mitigated_roc_auc": mit_eval.metrics.get("roc_auc", np.nan),
                # fairness (short names)
                "baseline_spd": base_fair.fairness_metrics["statistical_parity_difference"],
                "baseline_di": base_fair.fairness_metrics["disparate_impact"],
                "baseline_eod": base_fair.fairness_metrics["equal_opportunity_difference"],
                "baseline_aod": base_fair.fairness_metrics["average_odds_difference"],
                "baseline_ppv_diff": base_fair.fairness_metrics["predictive_parity_difference"],
                "mitigated_spd": mit_fair.fairness_metrics["statistical_parity_difference"],
                "mitigated_di": mit_fair.fairness_metrics["disparate_impact"],
                "mitigated_eod": mit_fair.fairness_metrics["equal_opportunity_difference"],
                "mitigated_aod": mit_fair.fairness_metrics["average_odds_difference"],
                "mitigated_ppv_diff": mit_fair.fairness_metrics["predictive_parity_difference"],
                "positive_rate": pos_rate,
            }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("threshold")

        # Save table artifacts
        csv_path = out_dir / "threshold_sweep_results.csv"
        json_path = out_dir / "threshold_sweep_results.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

        # Plot artifact
        plot_path = out_dir / "fairness_vs_threshold_di.png"
        _plot_di_vs_threshold(df, plot_path, di_min=di_min, di_max=di_max)

        # Recommendation + summary
        rec = _recommend_threshold(
            df,
            di_min=di_min,
            di_max=di_max,
            pos_min=pos_min,
            pos_max=pos_max,
            optimize_metric="mitigated_f1",
        )

        summary_path = out_dir / "recommended_threshold.md"
        summary_path.write_text(
            _summary_markdown(df, rec, di_min=di_min, di_max=di_max), encoding="utf-8"
        )

        # Log artifacts to MLflow
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(summary_path))

        # Log recommended values as MLflow metrics (easy to find in UI)
        mlflow.log_metrics(
            {
                "recommended_threshold": rec["recommended_threshold"],
                "recommended_mitigated_di": rec["recommended_mitigated_di"]
                if rec["recommended_mitigated_di"] is not None
                else np.nan,
                "recommended_mitigated_f1": rec["recommended_mitigated_f1"]
                if rec["recommended_mitigated_f1"] is not None
                else np.nan,
            }
        )

        out = {
            "run_id": run_id,
            "artifact_dir": str(out_dir),
            "recommendation": rec,
            "n_thresholds": int(len(thresholds)),
        }
        (out_dir / "sweep_run_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_dir / "sweep_run_summary.json"))

        return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Threshold sweep: fairness vs threshold + recommended threshold (DI constraint)."
    )
    p.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    p.add_argument("--min-threshold", type=float, default=100.0)
    p.add_argument("--max-threshold", type=float, default=800.0)
    p.add_argument("--step", type=float, default=50.0)
    p.add_argument("--di-min", type=float, default=0.8, help="Lower bound for DI constraint.")
    p.add_argument("--di-max", type=float, default=1.25, help="Upper bound for DI constraint.")
    p.add_argument("--test-size", type=float, default=BudgetConfig.test_size)
    p.add_argument("--random-state", type=int, default=BudgetConfig.random_state)
    p.add_argument(
        "--sample-n", type=int, default=None, help="Optional row sample size for faster sweeps."
    )
    p.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI.")
    p.add_argument(
        "--experiment-name", default=BudgetConfig.experiment_name, help="MLflow experiment name."
    )
    p.add_argument(
        "--pos-min", type=float, default=0.2, help="Minimum allowed positive rate P(y=1)."
    )
    p.add_argument(
        "--pos-max", type=float, default=0.8, help="Maximum allowed positive rate P(y=1)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = BudgetConfig(
        threshold=300.0,  # not used directly; labels are rebuilt per sweep threshold
        test_size=args.test_size,
        random_state=args.random_state,
        sample_n=args.sample_n,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )

    thresholds = _parse_thresholds(args.min_threshold, args.max_threshold, args.step)
    out = run_threshold_sweep(
        data_path=args.data_path,
        thresholds=thresholds,
        di_min=args.di_min,
        di_max=args.di_max,
        pos_min=args.pos_min,
        pos_max=args.pos_max,
        config=cfg,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

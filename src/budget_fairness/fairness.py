from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def _rate(condition: np.ndarray) -> float:
    # mean of boolean array = rate
    return float(np.mean(condition)) if condition.size else float("nan")


def _group_basic_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Returns:
      selection_rate = P(y_hat=1)
      tpr = P(y_hat=1 | y=1)
      fpr = P(y_hat=1 | y=0)
      ppv = P(y=1 | y_hat=1)   (precision)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    sel = _rate(y_pred == 1)

    positives = y_true == 1
    negatives = y_true == 0
    pred_pos = y_pred == 1

    tpr = (
        _rate((y_pred == 1) & positives) / _rate(positives)
        if _rate(positives) != 0
        else float("nan")
    )
    fpr = (
        _rate((y_pred == 1) & negatives) / _rate(negatives)
        if _rate(negatives) != 0
        else float("nan")
    )
    ppv = _rate(positives & pred_pos) / _rate(pred_pos) if _rate(pred_pos) != 0 else float("nan")

    return {
        "selection_rate": sel,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "ppv": float(ppv),
    }


@dataclass(frozen=True)
class FairnessResult:
    privileged_label: str
    unprivileged_label: str
    group_counts: dict[str, int]
    group_rates: dict[str, dict[str, float]]
    fairness_metrics: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "privileged_label": self.privileged_label,
            "unprivileged_label": self.unprivileged_label,
            "group_counts": self.group_counts,
            "group_rates": self.group_rates,
            "fairness_metrics": self.fairness_metrics,
        }


def compute_fairness_by_education(
    X: pd.DataFrame,
    y_true,
    y_pred,
    *,
    privileged_education_levels: tuple[str, ...] | list[str],
    education_col: str = "Education_Level",
) -> FairnessResult:
    """
    Computes fairness metrics using Education_Level as the sensitive attribute.

    Definitions (all for positive class y_hat=1):
      - Statistical Parity Difference (SPD) = SR_unpriv - SR_priv
      - Disparate Impact (DI) = SR_unpriv / SR_priv
      - Equal Opportunity Difference (EOD) = TPR_unpriv - TPR_priv
      - Average Odds Difference (AOD) = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
      - Predictive Parity Difference (PPV diff) = PPV_unpriv - PPV_priv
    """
    if education_col not in X.columns:
        raise ValueError(f"'{education_col}' column not found in X. Available: {list(X.columns)}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    edu = X[education_col].astype(str)
    priv_mask = edu.isin(list(privileged_education_levels)).to_numpy()
    unpriv_mask = ~priv_mask

    y_true_priv, y_pred_priv = y_true[priv_mask], y_pred[priv_mask]
    y_true_unpriv, y_pred_unpriv = y_true[unpriv_mask], y_pred[unpriv_mask]

    priv_rates = _group_basic_rates(y_true_priv, y_pred_priv)
    unpriv_rates = _group_basic_rates(y_true_unpriv, y_pred_unpriv)

    # Fairness metrics
    spd = unpriv_rates["selection_rate"] - priv_rates["selection_rate"]
    di = _safe_div(unpriv_rates["selection_rate"], priv_rates["selection_rate"])
    eod = unpriv_rates["tpr"] - priv_rates["tpr"]
    aod = 0.5 * (
        (unpriv_rates["fpr"] - priv_rates["fpr"]) + (unpriv_rates["tpr"] - priv_rates["tpr"])
    )
    ppv_diff = unpriv_rates["ppv"] - priv_rates["ppv"]

    fairness_metrics = {
        "statistical_parity_difference": float(spd),
        "disparate_impact": float(di),
        "equal_opportunity_difference": float(eod),
        "average_odds_difference": float(aod),
        "predictive_parity_difference": float(ppv_diff),
    }

    group_rates = {
        "privileged": priv_rates,
        "unprivileged": unpriv_rates,
    }
    group_counts = {
        "privileged": int(priv_mask.sum()),
        "unprivileged": int(unpriv_mask.sum()),
    }

    return FairnessResult(
        privileged_label=f"Education_Level in {tuple(privileged_education_levels)}",
        unprivileged_label=f"Education_Level not in {tuple(privileged_education_levels)}",
        group_counts=group_counts,
        group_rates=group_rates,
        fairness_metrics=fairness_metrics,
    )

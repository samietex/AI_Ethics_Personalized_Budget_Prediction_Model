from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    report: str


def evaluate_classifier(y_true, y_pred, y_prob=None) -> EvalResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            # e.g., if only one class present
            pass

    report = classification_report(y_true, y_pred, zero_division=0)
    return EvalResult(metrics=metrics, report=report)

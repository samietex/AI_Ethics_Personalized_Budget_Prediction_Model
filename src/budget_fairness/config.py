from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for training/evaluation.

    Notes:
      - threshold: used to binarize budget into `high_budget` label.
      - privileged_education_levels: used later for fairness reporting.
    """

    threshold: float = 300.0
    test_size: float = 0.2
    random_state: int = 42

    # Optional: downsample rows for fast iteration (None uses full dataset)
    sample_n: int | None = None

    # MLflow
    tracking_uri: str | None = None  # None -> MLflow default (./mlruns if run locally)
    experiment_name: str = "budget-fairness"
    registered_model_name: str = "budget_threshold_classifier"

    # Responsible AI (for later steps)
    privileged_education_levels: Sequence[str] = ("Bachelor’s Degree", "Master’s Degree")

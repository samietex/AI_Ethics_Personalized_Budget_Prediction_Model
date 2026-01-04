from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reweighing_weights(
    X: pd.DataFrame,
    y,
    *,
    privileged_education_levels: tuple[str, ...] | list[str],
    education_col: str = "Education_Level",
) -> np.ndarray:
    """
    Kamiran & Calders reweighing:
      w(y,a) = P(y) * P(a) / P(y,a)

    Where:
      y in {0,1} is the label
      a in {0,1} is the sensitive attribute (privileged=1, unprivileged=0)
    """
    if education_col not in X.columns:
        raise ValueError(f"'{education_col}' column not found in X. Available: {list(X.columns)}")

    y = np.asarray(y).astype(int)

    edu = X[education_col].astype(str)
    a = edu.isin(list(privileged_education_levels)).to_numpy().astype(int)  # privileged=1

    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    # Marginals
    p_y = {val: np.mean(y == val) for val in (0, 1)}
    p_a = {val: np.mean(a == val) for val in (0, 1)}

    # Joint
    p_ya = {}
    for yv in (0, 1):
        for av in (0, 1):
            p_ya[(yv, av)] = np.mean((y == yv) & (a == av))

    # Weights per instance
    w = np.zeros(n, dtype=float)
    for i in range(n):
        yv, av = int(y[i]), int(a[i])
        denom = p_ya[(yv, av)]
        w[i] = (p_y[yv] * p_a[av] / denom) if denom > 0 else 0.0

    return w

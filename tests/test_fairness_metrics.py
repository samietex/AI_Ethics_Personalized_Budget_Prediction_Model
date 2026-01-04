import pandas as pd

from budget_fairness.fairness import compute_fairness_by_education


def test_fairness_metrics_shapes_and_keys():
    X = pd.DataFrame(
        {
            "Age": [20, 30, 40, 50],
            "Gender": ["Male", "Female", "Female", "Male"],
            "Education_Level": ["Bachelor’s Degree", "High School Grad", "Bachelor’s Degree", "High School Grad"],
            "With children?": [0, 1, 0, 1],
            "Recommended_Activity": ["A", "B", "C", "D"],
        }
    )
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0]

    res = compute_fairness_by_education(
        X,
        y_true,
        y_pred,
        privileged_education_levels=("Bachelor’s Degree", "Master’s Degree"),
    )

    # basic sanity: keys exist
    assert "statistical_parity_difference" in res.fairness_metrics
    assert "disparate_impact" in res.fairness_metrics
    assert "equal_opportunity_difference" in res.fairness_metrics
    assert "average_odds_difference" in res.fairness_metrics
    assert "predictive_parity_difference" in res.fairness_metrics

    assert res.group_counts["privileged"] == 2
    assert res.group_counts["unprivileged"] == 2

import pandas as pd

from budget_fairness.mitigation import compute_reweighing_weights


def test_reweighing_weights_length_matches_rows():
    X = pd.DataFrame(
        {
            "Education_Level": [
                "Bachelor’s Degree",
                "High School Grad",
                "Master’s Degree",
                "High School Grad",
            ],
            "Age": [20, 30, 40, 50],
        }
    )
    y = [1, 0, 1, 0]
    w = compute_reweighing_weights(
        X, y, privileged_education_levels=("Bachelor’s Degree", "Master’s Degree")
    )
    assert len(w) == len(X)
    assert (w >= 0).all()

import pandas as pd

from budget_fairness.data import to_model_frame


def test_to_model_frame_creates_target():
    # Minimal frame with required columns
    df = pd.DataFrame(
        {
            "Budget (in dollars)": [100.0, 500.0],
            "Age": [20, 30],
            "Gender": ["Male", "Female"],
            "Education_Level": ["High School Grad", "Bachelor’s Degree"],
            "With children?": [0, 1],
            "Recommended_Activity": ["Stay in: Watch calming TV", "Play: Visit a movie theater"],
        }
    )
    X, y = to_model_frame(df, threshold=300)
    assert list(X.columns) == [c for c in df.columns if c != "Budget (in dollars)"]
    assert y.tolist() == [0, 1]

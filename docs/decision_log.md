# Decision Log

## Label definition
- `high_budget = 1` if `Budget (in dollars) >= 300`
- Rationale: aligns with project framing; can be swept later.

## Sensitive attribute choice
- Primary slicing: `Education_Level`
- Privileged group: Bachelor’s/Master’s
- Rationale: educational attainment used in original project and fairness evaluation.

## Model choice
- Logistic Regression baseline
- Rationale: interpretability, stable training, supports sample weights.

## Mitigation choice
- Reweighing (Kamiran & Calders)
- Rationale: simple, transparent, works well with LR via sample_weight.

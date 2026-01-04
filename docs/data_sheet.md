# Data Sheet — Budget Threshold Dataset (Project)

## Motivation
This dataset supports a learning project on Responsible AI: predicting budget threshold and evaluating fairness.

## Composition
- Unit of observation: an individual record with demographics + recommended activity + budget.
- Target label: derived from `Budget (in dollars)` using a threshold.

## Collection process
- Source: educational dataset used for Responsible AI learning.
- Not committed in this repo (local-only).

## Preprocessing
- Numeric imputation: median
- Categorical imputation: most frequent
- One-hot encoding for categoricals

## Recommended uses
- Model prototyping, fairness metric exploration, mitigation demonstration.

## Non-recommended uses
- High-stakes decisions affecting rights/opportunities without formal governance.

## Known limitations
- Unknown sampling/collection context → uncertain representativeness.
- Potential proxies for protected attributes.

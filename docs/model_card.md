# Model Card — Budget Threshold Classifier

## Model details
- **Model type:** Logistic Regression (scikit-learn Pipeline: preprocessing + classifier)
- **Task:** Binary classification — predict whether a user's budget is **>= threshold**
- **Default threshold (label rule):** $300
- **Mitigation implemented:** Reweighing (sample weights during training)
- **Latest training run id:** {{RUN_ID}}

## Intended use
**Primary use case**
- Support budget-aware personalization (e.g., recommending activities) by classifying likely high/low budget groups.

**Intended users**
- Data/ML practitioners and product stakeholders evaluating Responsible AI trade-offs.

**Out of scope**
- Any decisions that materially affect a person’s access to critical resources (credit, housing, employment) without additional governance, audits, and legal review.

## Training data
- Dataset: `udacity_ai_ethics_project_data.csv` (not committed to repo)
- Sensitive attribute used for fairness slicing: `Education_Level` (privileged vs unprivileged groups)
- Label: `high_budget = 1 if Budget (in dollars) >= threshold else 0`

## Evaluation metrics (test split)
### Baseline performance
- Accuracy: {{BASE_ACC}}
- F1: {{BASE_F1}}
- Precision: {{BASE_PREC}}
- Recall: {{BASE_REC}}
- ROC AUC: {{BASE_AUC}}

### Mitigated (reweighing) performance
- Accuracy: {{MIT_ACC}}
- F1: {{MIT_F1}}
- Precision: {{MIT_PREC}}
- Recall: {{MIT_REC}}
- ROC AUC: {{MIT_AUC}}

## Fairness evaluation (Education_Level slicing)
**Metrics**
- SPD (SR_unpriv - SR_priv)
- DI (SR_unpriv / SR_priv)
- EOD (TPR_unpriv - TPR_priv)
- AOD
- Predictive parity difference (PPV_unpriv - PPV_priv)

### Baseline fairness
- SPD: {{BASE_SPD}}
- DI: {{BASE_DI}}
- EOD: {{BASE_EOD}}
- AOD: {{BASE_AOD}}
- PPV diff: {{BASE_PPV}}

### Mitigated (reweighing) fairness
- SPD: {{MIT_SPD}}
- DI: {{MIT_DI}}
- EOD: {{MIT_EOD}}
- AOD: {{MIT_AOD}}
- PPV diff: {{MIT_PPV}}

## Key trade-offs (summary)
- DI changed from {{BASE_DI}} → {{MIT_DI}}
- F1 changed from {{BASE_F1}} → {{MIT_F1}}
- See: `reports/artifacts/comparison.md` and MLflow run artifacts.

## Ethical considerations
- **Proxy risk:** `Education_Level` may act as a proxy for socioeconomic status; outcomes can amplify inequality.
- **Measurement issues:** “Budget” may be noisy, self-reported, or context-dependent.
- **Allocation harm:** Misclassification could lead to exclusion from recommended opportunities or biased personalization.

## Limitations
- Dataset representativeness unknown; results may not generalize.
- Fairness evaluation is limited to one sensitive attribute and one grouping strategy.
- Fairness metrics do not fully capture downstream harms.

## Recommendations
- Monitor fairness metrics over time (drift + fairness regression).
- Expand subgroup analyses (intersectional slicing where appropriate).
- Consider alternative mitigations and threshold tuning with stakeholder constraints.

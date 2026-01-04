# Risk Assessment — Budget Threshold Classifier

## System overview
Binary classifier predicts whether a user's budget is >= threshold.

## Potential harms
### Allocation / quality-of-service harms
- Users may be recommended activities inconsistent with true budget.
- Disadvantaged groups may receive systematically different recommendations.

### Representational harms
- Reinforces stereotypes tied to education/socioeconomic status.

## Sensitive attributes & proxies
- Evaluated sensitive attribute: `Education_Level`
- Proxy risk: education can correlate with income, social class, and other protected characteristics.

## Fairness risks
- Disparate selection rates (DI/SPD)
- Differences in TPR (EOD) causing unequal benefit for true positives
- Differences in false positives (AOD/FPR differences)

## Mitigations implemented
- Reweighing to reduce dependence between sensitive group and label at training time.
- Fairness report artifacts logged each run.

## Residual risks
- Mitigation may shift error distribution or reduce overall performance.
- Fairness may regress under data drift.

## Monitoring plan (recommended)
- Track:
  - performance: accuracy/f1/roc_auc
  - fairness: DI, SPD, EOD, AOD, PPV diff
- Use:
  - MLflow runs for versioned audit trails
  - periodic evaluation job (later we’ll automate via CI)

## Deployment considerations
- Human oversight for any use beyond demo.
- Clear user communication: model is probabilistic and may be wrong.

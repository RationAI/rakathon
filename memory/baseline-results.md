---
name: baseline-results
description: Baseline model results - GB vs LR (~0.83-0.84 ROC AUC), history features add nothing, permutation importance, decision thresholds
metadata:
  type: project
---

All baselines are untuned sklearn models, one model per label. Untuned HistGradientBoosting reaches ROC AUC ~0.83-0.84 under every split tried.

**GB vs LR on the official patient-level split** (`training/baseline/`, Adam Džadoň's agent). Run from that folder: `uv run python compare.py` (or `gradient_boosting.py` / `logistic_regression.py`; shared code in `data.py`; root uv env). 19 raw features (all columns except IDs, INVITATION_DATE, labels).
- Gradient boosting: `HistGradientBoostingClassifier`, native categoricals and NaN.
- Logistic regression: one-hot categoricals (missing as own level), numeric NaN → 0 + missing indicator, log1p, standardise.
- On the stratified patient split (current official split):
  - GB test ROC AUC 0.841 / 0.839 / 0.841 (3M / 6M / 12M), PR AUC 0.292 / 0.438 / 0.581.
  - LR test ROC AUC 0.833 / 0.830 / 0.832, PR AUC 0.274 / 0.418 / 0.558.
  - GB wins every row, but the margins are small.

**Experiments on time splits A/B** (`features/*.py`, Adam Kukučka's agent; see [[split-decision]]):
- GB baseline: ROC AUC ~0.83, PR AUC 0.28 / 0.43 / 0.58.
- Adding 15 `H_*` history features: +0.001-0.002 AUC, so no real gain. `PRIOR_INVITATION_COUNT` / `MONTHS_SINCE_LAST_SCREEN` / `INVITATION_SEQUENCE` already carry this information, and women who responded are rarely re-invited.
- Permutation importance (12M, split B, AUC drop): PRIOR_INVITATION_COUNT 0.045, MONTHS_SINCE_LAST_SCREEN 0.040, INVITATION_SEQUENCE 0.014, GYN_PREVENTION_24M_CNT 0.010, HEALTHCARE_CONTACT_12M_CNT 0.007, OTHER_SCREENING_36M_CNT 0.007, MONTHS_LAST_INVITATION 0.006, region 0.003. Registration flags ~0 (redundant, not useless). Almost all `H_*` ~0.
- Confusion matrices (train 2020-21, F1-optimal thresholds chosen on 2022, tested on 2024): thresholds ~0.17 / 0.23 / 0.29 for 3M / 6M / 12M. The default 0.5 threshold is useless because base rates are low. 12M: specificity 0.879, NPV 0.903 (vs 0.808 share of negatives).

Related: [[feature-engineering]], [[open-items]].

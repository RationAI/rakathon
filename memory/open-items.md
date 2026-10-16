---
name: open-items
description: Combined to-do list and open questions from all agents (modelling, evaluation, questions for organisers)
metadata:
  type: project
---

**Modelling**
- Retrain the baseline on the prepared file (on the official split) to measure what spatial and engineered features add. Run logistic regression on the `LIN_*` feature set.
- Region target encoding fit on the train split only, inside the model pipeline.
- Enforce monotone predictions 3M ≤ 6M ≤ 12M. Calibrate the predictions.
- Choose decision thresholds on val (see [[pitfalls]]).

**Evaluation / split**
- Report test metrics per invitation year to measure period effects.
- Optional: stratify the split on each woman's fastest response (3M / 6M / 12M / never) instead of ever-screened.

**Questions for organisers**
- Exact definition of `PRIOR_INVITATION_COUNT` vs `INVITATION_SEQUENCE`.
- The missing pre-2020 invitation of the 52 women whose first row has sequence 2.

**Docs**
- `exploration/FINDINGS.md` still says `PRIOR_SCREEN_COUNT_5Y`=2 is likely a claim-code artefact; this is wrong (see [[dataset-facts]]).

**Not started:** tasks 2-4 (segmentation, interventions, experiment design).

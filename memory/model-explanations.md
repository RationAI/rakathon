---
name: model-explanations
description: What drives the classifiers - drawn depth-3 tree segments, shared top predictors, effect shapes (partial dependence), LR odds ratios
metadata:
  type: project
---

`training/classifiers/explain/` (`run.py` -> `results/`; `report.ipynb` / `report.md`). 12M label, default models, official split.

- Drawn trees: depth 3 (8 leaves) test ROC AUC 0.808, depth 4 0.822, real depth-8 tree 0.836. Depth-3 segments: <=2 prior invitations + GYN prevention visit + not screened in last ~12 months -> 57% screened (17% of invitations); >2 prior invitations + sequence >4 + no GYN prevention -> 4% (39% of invitations); screened in last ~12 months -> 3%.
- All models except naive Bayes share the same drivers: invitation history (PRIOR_INVITATION_COUNT / INVITATION_SEQUENCE), MONTHS_SINCE_LAST_SCREEN, GYN_PREVENTION_24M_CNT; then other screening, healthcare contact.
- Effect shapes: invitation count 23% -> 10-16% by 10th, flat after ~6; months since screen 2-10% at 0-1 months, peak 37-42% at 36, decline to ~31-33% at 59 (non-monotonic, LR cannot fit it); GYN prevention 0 -> 1 visit 16% -> 23% then flat; age modest 20-22% -> 17%; GYN registration flat once visits known.
- LR odds ratios: prior invitations 0.61, sequence 0.74, age 70-74 0.72; GYN prevention 1.33, other screening 1.26, contact 1.25. Area +-25% (Karviná 0.72 ... Kolín 1.25). Collinear encodings (registered flag vs missing band; months since screen vs its missing indicator) must be read as pairs.
- Practical: time invitations to ~3 years after last screen; the 39% chronic non-responder group without GYN prevention needs different interventions.

Related: [[model-ceiling]], [[per-group-experts]], [[baseline-results]].

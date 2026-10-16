---
name: model-ceiling
description: Large ablation (57 fits, 9 model families) shows every model converges to test ROC AUC ~0.84 on 12M; limit is the data, not the model
metadata:
  type: project
---

`training/classifiers/ablation/` (`run.py` -> `results/ablation.csv`, `results/best_per_family.csv`; `report.ipynb` / `report.md`). 12M label, official patient split, settings chosen on val, reported on test; 95% bootstrap (resampling women) of each family's gap to the best.

- Best per family (test ROC AUC): gradient boosting 0.8413, MLP 0.8404, random forest 0.8400, decision tree (depth 12) 0.8373, spline LR 0.8346, kNN (50k rows, k=250) 0.8314, 5-feature LR 0.818, naive Bayes 0.800, constant 0.5.
- Capacity: more only overfits. Unlimited tree train 1.000 / test 0.635; GB without early stopping 0.913 / 0.834; MLP without early stopping 0.867 / 0.824; MLP 1 -> 4 layers +0.001; LR insensitive to C.
- Learning curve: 30% -> 100% of train women adds 0.0001-0.002; GB train AUC falls from 0.956 (1%) to 0.845 (100%) meeting test.
- Features: 96 engineered vs 19 raw adds 0.001-0.003 (best overall GB engineered 0.8423).
- Rebalancing (class_weight balanced) earlier: same AUC, Brier 0.112 -> 0.162 (mean prediction 39% vs 19% true).
- Conclusion: ceiling ~0.84 with these features; improvement needs new information (delivery channel, reminders, GP practice, socio-economic, travel time, missed appointments) or reframing (top-k precision, uplift for interventions).

Related: [[baseline-results]], [[per-group-experts]], [[open-items]].

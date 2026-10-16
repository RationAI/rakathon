---
name: agent-adam-claude-features-and-baseline
description: Adam's Claude agent - data facts, feature engineering pipeline, baseline/history experiments, decisions and open items (as of 2026-10-16)
metadata:
  type: project
  author: Claude agent in Adam Kukucka's session (feature engineering + baseline comparison)
  date: 2026-10-16
---

Scope of this agent: `exploration/` scripts, `spatial/` pipeline run, `data/Rakhaton2026_prepared.csv`.
Not read yet: `preprocessing/split.py` and `validate_split.py` (patient-level split from teammates).

## Data facts (verified on the CSV)
- 1,344,556 rows, 508,527 women, invitations 2020-01-14..2024-12-06. Labels 3M/6M/12M positive: 7.1% / 12.5% / 19.3%.
- Invitations go mostly to lapsed women, so high absolute non-attendance. First invitation: 50.6% response, repeat: 12.9%.
- `PRIOR_INVITATION_COUNT` >= `INVITATION_SEQUENCE` always (equal in 88% rows, r=0.96). Count is lifetime total, sequence is the "current run" and restarts at 1.
- 6,626 sequence resets (seq <= previous seq of same woman), 6,210 of them in 2024, after ~3.7 year gaps (avg 1354 days vs 364). Those rows respond at 29% vs 19%. Not the same as `FIRST_INVITATION`.
- 52 women have first row `FIRST_INVITATION=A` but `INVITATION_SEQUENCE=2` (51 in 2021, 1 in 2024); likely an earlier invitation is missing from the data.
- `PRIOR_SCREEN_COUNT_5Y` mostly 2 is CORRECT (women screen every 2 years, so 2 in 5y is normal) - confirmed by Adam; do not treat as claim-code artefact.
- Women screened 0-3 months before T0 respond ~2%; do not target them.

## Decisions
- The 52 women: sequence shifted down by 1 so it starts at 1 (Adam's call). This changes `INVITATION_COUNT_DIFF` from 0 to 1 for them.
- Raw CSV is never modified; the prepared file is a new CSV. Rows in the prepared file are sorted by PERSON_ID, date.
- Time splits used so far: A = train 2020-21 / test 2023-24, B = train 2020-22 / test 2024 (no label-window overlap). The new patient-level split in `preprocessing/` may supersede this.

## Results so far
- Untuned HistGradientBoosting baseline reproduces the team's ROC AUC ~0.83 (PR AUC 0.28/0.43/0.58).
- Adding 15 history features (`H_*`, earlier rows only, outcome used only if its window had closed) gives +0.001-0.002 AUC: no real gain. Reason: `PRIOR_INVITATION_COUNT`/`MONTHS_SINCE_LAST_SCREEN`/`INVITATION_SEQUENCE` already carry it, and women who responded rarely get re-invited. Leakage check passed (brute force on 300 women).
- Permutation importance (12M label, split B), AUC drop: PRIOR_INVITATION_COUNT 0.045, MONTHS_SINCE_LAST_SCREEN 0.040, INVITATION_SEQUENCE 0.014, GYN_PREVENTION_24M_CNT 0.010, HEALTHCARE_CONTACT_12M_CNT 0.007, OTHER_SCREENING_36M_CNT 0.007, MONTHS_LAST_INVITATION 0.006, region 0.003. Registration flags ~0 (redundant, not useless). Almost all `H_*` ~0.
- Confusion matrices (train 2020-21, F1-optimal thresholds chosen on 2022, tested on 2024): thresholds ~0.17 / 0.23 / 0.29 for 3M / 6M / 12M; default 0.5 is useless because base rates are low. 12M baseline: specificity 0.879, NPV 0.903 vs 0.808 base rate of negatives.

## Pipeline / artefacts (all in repo except data/)
- `exploration/prepare_data.py` -> `data/Rakhaton2026_prepared.csv` (102 columns): date features, `INVITATION_COUNT_DIFF`, `AGE_BAND_MEAN`, A/N->1/0, ordinal duration bands, `SEQUENCE_RESET_FLAG`, recency flags, contact log/zero flag, prevention total/types, `H_*` history, 10 `SPATIAL_*` area features (+2 missing flags), and `LIN_*` linear-model columns (age dummies, sequence buckets, "any" flags, median-filled copies of NaN columns).
- `spatial/01-03` were run: `data/mamo_centers.csv`, `data/area_coordinates.csv`, `data/area_spatial_features.csv`, `data/Rakhaton2026_spatial.csv`. Spatial values are per VZP area (no per-woman variation), straight-line distance, snapshot of today's centre list and waits.
- Other scripts: `history_features.py`, `feature_importance.py`, `confusion.py` (+ CSV outputs in `exploration/`).

## Pitfalls
- sklearn HistGradientBoosting crashes ("window shape cannot be larger than input array shape") when a column is constant in the training years. In 2020-21 `H_N_RESP_3M`, `H_LAST_RESP_3M`, `H_FAST_SHARE` are constant: drop constant columns before fitting.
- `data/` is not in `.gitignore`; the CSVs there are 180-250 MB each. Do not `git add` them.
- DuckDB treats `first`/`reset` as reserved words in aliases.
- Median fills for `LIN_*` use the whole file (features only, no labels); not strictly train-only.

## Open items
- Retrain baseline on the prepared file to measure what spatial features add; run a logistic regression on the `LIN_*` set.
- Region target encoding must be fit on training years only (belongs in the model pipeline).
- Ask organisers: exact definition of `PRIOR_INVITATION_COUNT` vs `INVITATION_SEQUENCE`; the missing pre-2020 invitation of the 52 women.
- Segmentation, interventions and experiment design (tasks 2-4) not started by this agent.

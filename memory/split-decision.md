---
name: split-decision
description: Official data split is patient-level, stratified, 70/15/15 by PERSON_ID via sklearn train_test_split (seed 0); earlier time splits are superseded
metadata:
  type: project
---

**Official split** (`preprocessing/split.py`): women (`PERSON_ID`) split 70/15/15 with all their invitations, using sklearn `train_test_split` twice, `random_state=0`, stratified by "woman ever screened within 12M on any invitation" (~49.6% in each split). Input is the prepared dataset (`data/Rakhaton2026_prepared.csv`; flags/labels are 1/0, dates ISO). Output: `data/splits/{train,val,test}.csv`, values copied unchanged as text. Built by `preprocessing/pipeline.py`. Sizes ≈ train 941,822 / val 200,883 / test 201,851 rows. `preprocessing/validate_split.py` checks it (no shared women, full coverage, identical to a fresh split, etc.).

**Why:** Adam Džadoň decided patient-level leakage is more serious than time leakage. Rejected alternatives:
1. Out-of-time split (train 2020-21, 365-day embargo, val 2023, test 2024): 62-78% of women overlapped between splits.
2. "Priority" time+person split: biased labels badly (train 44%, val 72% response) because screened women stop being re-invited.
3. Hand-written seeded shuffle: too complicated; a plain stratified split was wanted.

**Superseded:** Adam Kukučka's agent used two time splits in `features/` scripts: A = train 2020-21 / test 2023-24, B = train 2020-22 / test 2024 (no label-window overlap). Results from those (see [[baseline-results]]) are not directly comparable with the official split.

**Remaining leakage with the patient split:** period effects (all splits span 2020-2024, e.g. COVID 2020) and shared invitation batches. Suggested but not done: report test metrics per invitation year.

**How to apply:** use `data/splits/` for new experiments; don't reintroduce time-based splitting without asking. Any fitted preprocessing (median fills, target encoding) must be fit on the train split only. Related: [[pitfalls]], [[open-items]].

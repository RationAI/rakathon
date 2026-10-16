---
name: per-group-experts
description: Per-group expert models never beat the global GB model; hardest groups are repeat non-responders; drivers differ by group (timing vs prevention vs invitation history)
metadata:
  type: project
---

Experiment in `training/per_group_experts/` (`experts.py` writes `results/group_scores.csv` + `group_importance.csv`; `report.ipynb` / `report.md` present them). 12M label, 19 original features, untuned HistGradientBoosting, official patient split, evaluated on test. Expert = model trained on one group's train rows; compared with the global model on that group's test rows; 95% bootstrap interval of the AUC gain resampling women.

Criteria: age (<50 / 50-64 / 65+), first invitation, screening history (never in 5y / before), invitation sequence (1 / 2-3 / 4-6 / 7+), gynaecologist registered, healthcare contact 12m (0 / 1-10 / 11+), distance to nearest centre (<=10 / 10-25 / >25 km), region (14 kraje; VZP areas mapped via the geocoder's `matched` field in `data/area_coordinates.csv`, Praha-východ/západ belong to Středočeský kraj).

Results:
- Experts never beat the global model in any of 33 groups (gain -0.017..+0.000). Biggest losses in small/rare groups (0 contacts -0.017, no gynaecologist -0.007, seq 4-6 -0.006). Decision: keep one global model; group splits are for insight only.
- Hardest groups (within-group AUC): seq 4-6 0.67, seq 7+ 0.68 (36% of train rows, 4% response), screened before 0.75, first invitation 0.76, no contact 0.76. Easiest: never screened 0.85, >25 km 0.85.
- Drivers: young / first invitation -> prevention behaviour (GYN prevention 22-28%, other screening 13-15%); 50+ and screened before -> timing (MONTHS_SINCE_LAST_SCREEN 52-66%); never screened / no gyn / no contact -> invitation history (seq + prior count 59-66%); 7+ -> no strong signal.
- Distance to centre barely changes response (19.4 / 19.3 / 18.5%).
- Regions: experts never help (gain -0.006..-0.000); same drivers everywhere. Location is a baseline shift: lowest response Moravskoslezský 16.7%, Olomoucký 17.5% (areas Karviná 12.8%, Ostrava 15.0%), highest Vysočina 21.1% (Pelhřimov 23.8%). Praha hardest to predict (AUC 0.818). Report has a map of response rate per area.

Related: [[baseline-results]], [[open-items]].

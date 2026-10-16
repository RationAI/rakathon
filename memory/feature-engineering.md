---
name: feature-engineering
description: Prepared-data pipeline (features/prepare_data.py, 102 columns), spatial area features, history features and their decisions
metadata:
  type: project
---

Built by Adam Kukučka's agent. The raw CSV is never modified; derived data goes to new files in `data/` (gitignored, so each person must regenerate them locally).

**`features/prepare_data.py` → `data/Rakhaton2026_prepared.csv`** (102 columns, rows sorted by PERSON_ID, date):
- date features, `INVITATION_COUNT_DIFF`, `AGE_BAND_MEAN`, A/N → 1/0, ordinal duration bands, `SEQUENCE_RESET_FLAG`, recency flags, contact log / zero flag, prevention total / types,
- `H_*` history features (earlier rows of the same woman only; an outcome is used only if its label window had closed),
- 10 `SPATIAL_*` area features + 2 missing flags,
- `LIN_*` linear-model columns: age dummies, sequence buckets, "any" flags, median-filled copies of NaN columns.

**Bug fixed (2026-10-16):** `prepare_data.py` used to write `INVITATION_DATE` from a copy taken before re-sorting the rows, so 95.8% of rows got another row's date. Fixed; verified 100% match with the raw file. Any `Rakhaton2026_prepared.csv` built before this fix has wrong dates and must be rebuilt (`preprocessing/pipeline.py`).

**Decision:** for the 52 women whose first row has `INVITATION_SEQUENCE`=2, the sequence is shifted down by 1 so it starts at 1 (Adam Kukučka's call). This changes `INVITATION_COUNT_DIFF` from 0 to 1 for them. Only their first run of invitations is shifted (148 rows): 3 of them later restart at 1 after a long break, and shifting those rows too had produced sequence 0 (fixed 2026-10-17; `prepare_data.py` now asserts sequence >= 1).

**Spatial pipeline** (`features/spatial/01_scrape_centers.py`, `02_geocode_areas.py`, `03_add_spatial_features.py`; see `features/spatial/README.md`) produces `data/mamo_centers.csv`, `data/area_coordinates.csv`, `data/area_spatial_features.csv`, `data/Rakhaton2026_spatial.csv`. Values are per VZP territorial area (no per-woman variation), use straight-line distance, and are a snapshot of today's centre list and waiting times (not historical).

**History features** (`features/history_features.py`): 15 `H_*` features gave only +0.001-0.002 AUC. A brute-force leakage check on 300 women passed. See [[baseline-results]].

Related: [[dataset-facts]], [[pitfalls]], [[open-items]].

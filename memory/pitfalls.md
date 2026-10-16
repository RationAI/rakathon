---
name: pitfalls
description: Known traps when working with this data and code (constant columns in HGB, leakage in fitted preprocessing, DuckDB keywords, big files)
metadata:
  type: project
---

- **HistGradientBoosting crash** "window shape cannot be larger than input array shape" when a column is constant in the training data. In 2020-21, `H_N_RESP_3M`, `H_LAST_RESP_3M`, `H_FAST_SHARE` are constant: drop constant columns before fitting.
- **Fit preprocessing on train only.** The `LIN_*` median fills in `prepare_data.py` use the whole file (features only, no labels), which is not strictly train-only. Region target encoding must be fit on the train split only.
- **Index alignment:** assigning a Series saved before `sort_values(...).reset_index()` puts values on the wrong rows (this broke dates in `prepare_data.py`). Recompute from the current frame instead.
- **Large files:** CSVs in `data/` are 180-250 MB. `data/` is gitignored; never force-add them.
- **DuckDB** treats `first` / `reset` as reserved words in aliases.
- **IDs:** read `PERSON_ID` as a string or the leading zeros are lost. Writing floats back to CSV turns `12` into `12.0`; read as `dtype=str` when copying values through.
- **Labels are nested:** independent per-label models can predict 3M > 6M; enforce monotonicity if it matters.
- **Thresholds:** with base rates of 7-19%, the default 0.5 threshold is useless; choose thresholds on validation data.

Related: [[feature-engineering]], [[split-decision]], [[baseline-results]].

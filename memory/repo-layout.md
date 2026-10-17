---
name: repo-layout
description: Intended repository structure (single root uv project, folder purposes, rules) and how to run things
metadata:
  type: reference
---

The intended structure is documented in the root `README.md` (section "Repository structure"); keep both in sync.

**One uv project** for the whole repo: root `pyproject.toml` + `uv.lock` + `.python-version` (3.12) + `.venv/`. No per-folder pyprojects or venvs. Add dependencies with `uv add <pkg>` from the root. `.ruff.toml` at the root is the lint/format config for all code (its `src` lists the script folders so sibling imports count as first-party).

Folders:
- `data/` (gitignored): raw `Rakhaton2026.csv`, spec docx, `splits/`, and generated files (`*_prepared.csv`, spatial CSVs; see [[feature-engineering]]).
- `memory/`: this shared memory.
- `exploration/`: EDA only. `eda.ipynb` (committed with outputs), `eda.md` + `eda_files/` (export: `uv run jupyter nbconvert --to markdown eda.ipynb`), `FINDINGS.md`, `label_correlations.ipynb` (+ `.md`; attribute-label correlations on the prepared data), `covid.ipynb` (+ `.md`; COVID effects), `exploration.html` (whole exploration as one interactive page, built by `build_exploration.py` from `exploration.template.html` in the style of Jakub's `response_report.html`), `classifiers.html` ("Predictive models": model descriptions, saturation curves, attributes across models, segments, interventions, experiment, per-group experts; `build_classifiers.py` fits GB once for the risk deciles, ~1 min), Jakub's `response_report.html` (`build_report.py`; the separate map page `response_map.html` was removed as redundant). All three HTML pages are English, red VZP theme, and share a top `<nav class="site">` (edit it in every template); rebuild with `PYTHONUTF8=1 uv run python build_*.py` from `exploration/`. `playground.ipynb` (Adam Džadoň's scratch notebook, gitignored).
- `features/`: `prepare_data.py`, `history_features.py`, `feature_importance.py`, `confusion.py` (they import each other by module name, so they stay together), their result CSVs, and `spatial/` (scripts 01-03 + README).
- `segmentation/`: `segments.py` (`assign(df)` -> 8 rule-based segments A-H) + `report.ipynb` / `report.md`; see [[segments]].
- `presentation/`: `make_assets.py` (Czech charts -> `assets/*.png` + `assets/cisla.json`), `deck/project/` (slide source); see [[presentation]].
- `preprocessing/`: `pipeline.py` (single entry point that rebuilds all derived data), `split.py`, `validate_split.py`.
- `training/`: `classifiers/` (renamed from `baseline/`; `data.py` shared incl. `scaled_one_hot` preprocessing + `compare.py`; each model in its own package folder (`gradient_boosting/`, `logistic_regression/`, `spline_logistic_regression/`, `decision_tree/`, `random_forest/`, `mlp/`, `knn/`, `naive_bayes/`) with `model.py` exposing `build(x, **settings)` and `make_model`; run one with `uv run python -m <model>.model` from `training/classifiers`; `ablation/run.py` + `ablation/report.ipynb` = large ablation, see [[model-ceiling]]; `explain/run.py` + `explain/report.ipynb` = model explanations, see [[model-explanations]]; extra scripts: `decision_tree/tuned.py`, `mlp/neural_network.py` (PyTorch multi-output), `gradient_boosting/segmented.py` (Jakub Pekár's global vs per-segment GB)) and `per_group_experts/` (per-group models + `report.ipynb`; see [[per-group-experts]]). The Lightning template (`ml/`, `configs/`) was removed.

Rules: data never in git and always regenerable by a script; notebooks for exploration, scripts for reusable code; run each script from its own folder (they use paths relative to their folder, e.g. `../data`).

Commands: `cd preprocessing && uv run python pipeline.py` (spatial 01-03 if missing -> prepare_data -> split -> validate); `cd training/classifiers && uv run python compare.py`; `cd exploration && uv run jupyter lab`; lint `uv run ruff check <folder>` from the root (rejects uppercase argument names like `X`; `fix = true` rewrites files). Related: [[user-preferences]], [[pitfalls]].

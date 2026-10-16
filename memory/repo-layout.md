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
- `exploration/`: EDA only. `eda.ipynb` (committed with outputs), `eda.md` + `eda_files/` (export: `uv run jupyter nbconvert --to markdown eda.ipynb`), `FINDINGS.md`, `playground.ipynb` (Adam Džadoň's scratch notebook, gitignored).
- `features/`: `prepare_data.py`, `history_features.py`, `feature_importance.py`, `confusion.py` (they import each other by module name, so they stay together), their result CSVs, and `spatial/` (scripts 01-03 + README).
- `preprocessing/`: `pipeline.py` (single entry point that rebuilds all derived data), `split.py`, `validate_split.py`.
- `training/`: `baseline/` (sklearn GB vs LR), `per_group_experts/` (per-group models + `report.ipynb`; see [[per-group-experts]]), `ml/` (Lightning template package) and `configs/` (Hydra).

Rules: data never in git and always regenerable by a script; notebooks for exploration, scripts for reusable code; run each script from its own folder (they use paths relative to their folder, e.g. `../data`).

Commands: `cd preprocessing && uv run python pipeline.py` (spatial 01-03 if missing -> prepare_data -> split -> validate); `cd training/baseline && uv run python compare.py`; `cd exploration && uv run jupyter lab`; lint `uv run ruff check <folder>` from the root (rejects uppercase argument names like `X`; `fix = true` rewrites files). Related: [[user-preferences]], [[pitfalls]].

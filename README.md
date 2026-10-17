# Rakathon 2026

Predicting which women respond to VZP's addressed invitations to mammography screening.
One row of the dataset is one invitation; the model predicts whether the woman is screened
within 3, 6 and 12 months (`SCREENING_{3,6,12}M_FLAG`).

## Setup

One [uv](https://docs.astral.sh/uv/) project for the whole repository (Python 3.12):

```sh
uv sync                 # creates .venv/ in the repo root
```

Put the raw data into `data/` (not in git): `data/Rakhaton2026.csv` and the spec
`data/Rakathon_Datove_rozhrani.docx`.

Scripts take no command-line arguments and read/write `data/` relative to their own
folder, so **run each script from the folder it lives in** (`uv run` finds the root
project from any subfolder).

## Repository structure

```
rakathon/
├── pyproject.toml, uv.lock     the single uv project (all dependencies)
├── .python-version             3.12
├── .ruff.toml                  lint/format config for all Python code
├── data/                       raw data + everything generated from it (gitignored)
│   ├── Rakhaton2026_prepared.csv  fixed + feature-engineered dataset (prepare_data.py)
│   └── splits/                 train/val/test CSVs of the prepared dataset (split.py)
├── memory/                     shared notes for people and AI agents (start at MEMORY.md)
├── exploration/                EDA only: notebooks and their reports
│   ├── eda.ipynb               main EDA (committed with outputs)
│   ├── eda.md, eda_files/      markdown export of eda.ipynb
│   ├── FINDINGS.md             summary of the EDA
│   ├── label_correlations.ipynb  correlation of every attribute with the 3 labels (+ .md export)
│   ├── covid.ipynb             how COVID-19 shows in the data and the model (+ .md export)
│   ├── exploration.html        the whole exploration as one interactive page (build_exploration.py)
│   ├── classifiers.html        predictive models, segments, interventions, experiment (build_classifiers.py)
│   ├── response_report.html    regional report: district maps, access, waits, population (build_report.py)
│                               the three pages share a top navigation; *.template.html + build_*.py
│   └── playground.ipynb        personal scratch notebook (gitignored)
├── features/                   feature engineering and feature experiments
│   ├── prepare_data.py         raw CSV -> data/Rakhaton2026_prepared.csv (102 columns)
│   ├── history_features.py     per-woman history features (+ comparison)
│   ├── feature_importance.py   permutation importance
│   ├── confusion.py            confusion matrices / thresholds
│   ├── *.csv                   small result tables written by the scripts above
│   └── spatial/                area-level spatial features (run 01 -> 02 -> 03, see README)
├── segmentation/               explainable groups of invited women for interventions
│   ├── segments.py             8 rule-based segments (assign() reusable for later tasks)
│   ├── tree_diagram.py         draws the rules as a decision tree -> segment_tree.png
│   └── report.ipynb / .md      profiles, stability, regions, suggested interventions
├── presentation/               Czech presentation for the Rakathon jury
│   ├── make_assets.py          builds every chart as PNG + cisla.json (all quoted numbers)
│   ├── assets/                 the charts
│   └── deck/project/           source of the slide deck (16 slides, tasks 01-04)
├── preprocessing/              data pipeline
│   ├── pipeline.py             ONE command: spatial -> prepare_data -> split -> validate
│   ├── split.py                patient-level stratified 70/15/15 split
│   └── validate_split.py       checks the split files
└── training/                   models
    ├── classifiers/            one folder per model; data.py (shared) + compare.py (all models)
    │   │                         run one from classifiers/: uv run python -m <folder>.model
    │   ├── gradient_boosting/    model.py + segmented.py (global vs per-segment GB -> segmented_results.csv)
    │   ├── logistic_regression/  model.py
    │   ├── spline_logistic_regression/  model.py
    │   ├── decision_tree/        model.py + tuned.py (depth/leaf search, prints the tree as rules)
    │   ├── random_forest/        model.py
    │   ├── mlp/                  model.py (sklearn) + neural_network.py (PyTorch, 3 outputs at once)
    │   ├── knn/                  model.py (fits on a 50k-row sample)
    │   ├── naive_bayes/          model.py
    │   ├── explain/              run.py: drawn trees, permutation importance, partial dependence,
    │   │                         LR odds ratios -> results/; report.ipynb / report.md presents them
    │   └── ablation/             run.py (57 fits: capacity, learning curve, features) ->
    │                             results/*.csv; report.ipynb / report.md presents them
    └── per_group_experts/      one model per group (age, first invitation, ...) vs the global model;
                                experts.py -> results/*.csv, report.ipynb / report.md presents them
```

Rules for keeping it clean:
- Data never goes into git; everything in `data/` must be reproducible by a script.
- New exploration goes into `exploration/` (notebooks); reusable code goes into a
  script folder (`features/`, `preprocessing/`, `training/`).
- No per-folder `pyproject.toml` or virtualenv; add dependencies to the root with
  `uv add <package>`.

## Common commands

```sh
cd preprocessing && uv run python pipeline.py        # rebuild all derived data + splits
cd training/classifiers && uv run python compare.py  # all classifiers on the split
cd exploration && uv run jupyter lab                 # notebooks
uv run ruff check <folder>                           # lint (from repo root)
```

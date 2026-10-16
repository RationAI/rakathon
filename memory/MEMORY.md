# Shared agent memory

Merged from the agents of Adam Džadoň (EDA, split, baselines) and Adam Kukučka (feature engineering, spatial, history features).

- [Task definition](project-task-definition.md) — invitation predictor: one row = one invitation, predict 3M/6M/12M screening flags; team and agents
- [Dataset facts](dataset-facts.md) — size, A/N flags, string PERSON_ID, nested labels, structural NaNs, sequence resets, column quirks
- [Split decision](split-decision.md) — official patient-level stratified 70/15/15 split (seed 0); superseded time splits
- [Feature engineering](feature-engineering.md) — prepared 102-column file, spatial area features, history features
- [Baseline results](baseline-results.md) — GB vs LR ~0.83-0.84 ROC AUC, history features add nothing, importance, thresholds
- [Pitfalls](pitfalls.md) — HGB constant-column crash, train-only fitting, big files, DuckDB keywords
- [Open items](open-items.md) — combined to-do list and questions for organisers
- [Repo layout](repo-layout.md) — intended structure (single root uv project), folder rules, how to run / lint
- [User preferences](user-preferences.md) — Adam Džadoň: keep it simple, no CLI args, uv, CSV

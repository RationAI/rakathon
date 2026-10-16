"""Random-forest baseline: one RandomForestClassifier per label.

100 trees, each grown on a random 25% of the rows (keeps deep trees fast and in
memory). Default depth 16 with at least 20 rows per leaf: the best setting on val in
the ablation; fully grown trees overfit. Categorical columns are ordinal-encoded
(see ``data.ordinal_encoded``).

Usage:
    uv run python -m random_forest.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from data import SEED, ordinal_encoded, run


def build(
    _x: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int | None = 16,
    min_samples_leaf: int = 20,
    max_samples: float = 0.25,
) -> Pipeline:
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=SEED,
    )
    return make_pipeline(ordinal_encoded(), forest)


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Random forest")

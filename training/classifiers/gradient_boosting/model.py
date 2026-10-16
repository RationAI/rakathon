"""Gradient-boosting baseline: one untuned HistGradientBoostingClassifier per label.

The trees handle categorical columns and NaN natively, so features go in as they are.

Usage:
    uv run python -m gradient_boosting.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from data import SEED, run


def build(_x: pd.DataFrame, **params: object) -> HistGradientBoostingClassifier:
    """Sklearn defaults unless overridden (e.g. ``max_leaf_nodes``, ``max_iter``)."""
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=SEED, **params
    )


def make_model(x: pd.DataFrame) -> HistGradientBoostingClassifier:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Gradient boosting")

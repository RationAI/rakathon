"""Gradient-boosting baseline: one untuned HistGradientBoostingClassifier per label.

The trees handle categorical columns and NaN natively, so features go in as they are.

Usage:
    uv run python gradient_boosting.py
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from data import SEED, run


def make_model(_x: pd.DataFrame) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=SEED
    )


if __name__ == "__main__":
    run(make_model, "Gradient boosting")

"""Logistic-regression baseline: one L2-regularised model per label.

Input is one-hot encoded, imputed, log-transformed and standardised
(see ``data.scaled_one_hot``).

Usage:
    uv run python -m logistic_regression.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from data import SEED, run, scaled_one_hot


def build(x: pd.DataFrame, c: float = 1.0) -> Pipeline:
    """``c``: inverse regularisation strength (smaller = stronger L2 penalty)."""
    return make_pipeline(
        scaled_one_hot(x), LogisticRegression(C=c, max_iter=1000, random_state=SEED)
    )


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Logistic regression")

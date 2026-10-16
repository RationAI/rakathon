"""Naive-Bayes baseline: one Gaussian naive Bayes per label.

The simplest probabilistic model: every feature is treated as independent given the
label. Input is scaled and one-hot encoded (see ``data.scaled_one_hot``), then made
dense. ``var_smoothing`` 1e-2 was best on val in the ablation (sklearn's default 1e-9
is clearly worse on the one-hot columns).

Usage:
    uv run python -m naive_bayes.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from data import run, scaled_one_hot, to_dense


def build(x: pd.DataFrame, var_smoothing: float = 1e-2) -> Pipeline:
    return make_pipeline(
        scaled_one_hot(x),
        FunctionTransformer(to_dense, accept_sparse=True),
        GaussianNB(var_smoothing=var_smoothing),
    )


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Naive Bayes")

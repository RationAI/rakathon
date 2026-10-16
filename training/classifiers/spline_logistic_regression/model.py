"""Spline logistic regression: an additive model, one per label.

Each numeric feature is expanded into cubic splines, so the model can learn a smooth
non-linear curve per feature (e.g. the rise and fall of response with months since
the last screen), but there are still no interactions between features. Categorical
columns are one-hot encoded and flags pass through, as in plain logistic regression.

Usage:
    uv run python -m spline_logistic_regression.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer

from data import CATEGORICAL, FLAGS, SEED, numeric_features, run, signed_log1p


def build(x: pd.DataFrame, n_knots: int = 6) -> Pipeline:
    numeric = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        FunctionTransformer(signed_log1p),
        SplineTransformer(n_knots=n_knots, degree=3),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features(x)),
            ("categorical", categorical, CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ]
    )
    return make_pipeline(
        preprocess, LogisticRegression(max_iter=2000, random_state=SEED)
    )


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Spline logistic regression")

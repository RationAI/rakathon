"""Logistic-regression baseline: one L2-regularised model per label.

Preprocessing a linear model needs:
- categorical columns are one-hot encoded, with missing ("not registered") as its own
  level and unseen levels ignored,
- numeric NaNs (no previous invitation / screening) are filled with 0 plus a
  missing indicator,
- numeric columns are log1p-transformed (counts are long-tailed) and standardised.

Usage:
    uv run python logistic_regression.py
"""

import numpy as np
import pandas as pd
from data import CATEGORICAL, FLAGS, SEED, numeric_features, run
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def make_model(x: pd.DataFrame) -> Pipeline:
    numeric = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        FunctionTransformer(np.log1p),
        StandardScaler(),
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
        preprocess, LogisticRegression(max_iter=1000, random_state=SEED)
    )


if __name__ == "__main__":
    run(make_model, "Logistic regression")

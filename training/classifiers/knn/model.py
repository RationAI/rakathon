"""k-nearest-neighbours baseline: one kNN per label.

Prediction cost grows with (training rows x predicted rows), so the model keeps a
random sample of ``max_rows`` training rows and predicts in chunks. k = 250 on 50k
rows was best on val in the ablation; small k overfits (k = 5: train 0.91, test 0.76
ROC AUC). Input is scaled and one-hot encoded (see ``data.scaled_one_hot``).

Usage:
    uv run python -m knn.model    (from training/classifiers)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from data import SEED, run, scaled_one_hot, to_dense


class SampledKNN(ClassifierMixin, BaseEstimator):
    """kNN fitted on a random row sample, predicting in chunks to bound memory."""

    def __init__(
        self, k: int = 250, max_rows: int = 50_000, chunk: int = 50_000
    ) -> None:
        self.k = k
        self.max_rows = max_rows
        self.chunk = chunk

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SampledKNN":
        rows = np.arange(len(x))
        if len(x) > self.max_rows:
            rng = np.random.default_rng(SEED)
            rows = rng.choice(rows, self.max_rows, replace=False)
        self.model_ = make_pipeline(
            scaled_one_hot(x),
            FunctionTransformer(to_dense, accept_sparse=True),
            KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1),
        ).fit(x.iloc[rows], y.iloc[rows])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        parts = [
            self.model_.predict_proba(x.iloc[i : i + self.chunk])
            for i in range(0, len(x), self.chunk)
        ]
        return np.concatenate(parts)


def build(_x: pd.DataFrame, k: int = 250, max_rows: int = 50_000) -> SampledKNN:
    return SampledKNN(k=k, max_rows=max_rows)


def make_model(x: pd.DataFrame) -> SampledKNN:
    return build(x)


if __name__ == "__main__":
    run(make_model, "kNN (k=250, 50k training rows)")

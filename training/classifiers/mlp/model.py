"""MLP baseline: one multi-layer perceptron per label.

sklearn ``MLPClassifier`` (ReLU, Adam) on the same scaled, one-hot input as logistic
regression (see ``data.scaled_one_hot``). Early stopping on a 10% hold-out of the
train rows keeps it from overfitting. ``ablation/run.py`` varies depth and width.

Usage:
    uv run python -m mlp.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from data import SEED, run, scaled_one_hot


HIDDEN_LAYERS = (256, 128, 64)


def build(
    x: pd.DataFrame,
    hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
    alpha: float = 1e-4,
    early_stopping: bool = True,
    max_iter: int = 50,
) -> Pipeline:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=SEED,
    )
    return make_pipeline(scaled_one_hot(x), mlp)


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x, HIDDEN_LAYERS)


if __name__ == "__main__":
    run(make_model, f"MLP {HIDDEN_LAYERS}")

"""Decision-tree baseline: one depth-limited DecisionTreeClassifier per label.

A single tree is the most interpretable baseline: its splits read as rules. Depth and
leaf size are limited so the tree stays readable and does not memorise the training
data. Categorical columns are ordinal-encoded (see ``data.ordinal_encoded``).

Usage:
    uv run python -m decision_tree.model    (from training/classifiers)
"""

import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier

from data import SEED, ordinal_encoded, run


def build(
    _x: pd.DataFrame, max_depth: int | None = 8, min_samples_leaf: int = 500
) -> Pipeline:
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED
    )
    return make_pipeline(ordinal_encoded(), tree)


def make_model(x: pd.DataFrame) -> Pipeline:
    return build(x)


if __name__ == "__main__":
    run(make_model, "Decision tree")

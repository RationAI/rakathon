"""Compare the baselines side by side on the same splits.

Loads the data once, evaluates every model in ``MODELS`` and prints one table per
metric with a column per model. For ROC AUC and PR AUC higher is better, for Brier
lower is better; the ``best`` column names the winner of each row.

Usage (from training/classifiers):
    uv run python compare.py
"""

import pandas as pd

from data import ModelFactory, evaluate, prepare
from decision_tree import model as decision_tree
from gradient_boosting import model as gradient_boosting
from knn import model as knn
from logistic_regression import model as logistic_regression
from mlp import model as mlp
from naive_bayes import model as naive_bayes
from random_forest import model as random_forest
from spline_logistic_regression import model as spline_logistic_regression


# Each model lives in its own folder (<name>/model.py) and exposes make_model.
MODELS: dict[str, ModelFactory] = {
    "gradient boosting": gradient_boosting.make_model,
    "logistic regression": logistic_regression.make_model,
    "spline logistic regression": spline_logistic_regression.make_model,
    "decision tree": decision_tree.make_model,
    "random forest": random_forest.make_model,
    "MLP": mlp.make_model,
    "kNN": knn.make_model,
    "naive Bayes": naive_bayes.make_model,
}
METRICS = {"ROC AUC": "max", "PR AUC": "max", "Brier": "min"}


def main() -> None:
    data, X = prepare()
    print(f"Train rows: {len(data['train']):,} | features: {X['train'].shape[1]}")

    results = {}
    for name, make_model in MODELS.items():
        print(f"Fitting {name}...")
        results[name] = evaluate(make_model, data, X)

    for metric, best in METRICS.items():
        table = pd.DataFrame({name: r[metric] for name, r in results.items()})
        table["best"] = table.idxmax(axis=1) if best == "max" else table.idxmin(axis=1)
        arrow = "higher" if best == "max" else "lower"
        print(f"\n{metric} ({arrow} is better)")
        print(table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()

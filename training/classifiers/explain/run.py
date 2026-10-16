"""Explain the classifiers: what drives their predictions?

12M label, official split, every model in its default setting (as in ``compare.py``).
Writes to ``results/``:

- ``models.csv``: test ROC AUC of each fitted model,
- ``permutation_importance.csv``: drop in test ROC AUC when one feature is shuffled
  (20k test rows, 5 repeats), for every model,
- ``partial_dependence.csv``: average predicted response when one feature is set to a
  value for everyone (4k train rows; ``MONTHS_SINCE_LAST_SCREEN`` only over women
  screened before, where it exists),
- ``lr_coefficients.csv``: logistic-regression coefficients and odds ratios,
- ``tree_importance.csv``: impurity importance of the decision tree and random forest,
- ``tree_depth{3,4}.png`` + ``tree_rules.txt``: small readable trees drawn, and the top
  levels of the real (depth-8) decision tree as text,
- ``category_codes.csv``: what the integer codes in the tree splits mean.

Usage (from training/classifiers, ~15 min):
    uv run python -m explain.run
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text, plot_tree

from data import CATEGORICAL, FEATURES, SEED, prepare
from decision_tree import model as decision_tree
from gradient_boosting import model as gradient_boosting
from knn import model as knn
from logistic_regression import model as logistic_regression
from mlp import model as mlp
from naive_bayes import model as naive_bayes
from random_forest import model as random_forest
from spline_logistic_regression import model as spline_logistic_regression


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

LABEL = "SCREENING_12M_FLAG"
RESULTS = Path(__file__).resolve().parent / "results"
IMPORTANCE_ROWS = 20_000
PD_ROWS = 4_000

MODELS = {
    "gradient boosting": gradient_boosting.make_model,
    "random forest": random_forest.make_model,
    "MLP": mlp.make_model,
    "decision tree": decision_tree.make_model,
    "spline logistic regression": spline_logistic_regression.make_model,
    "logistic regression": logistic_regression.make_model,
    "kNN": knn.make_model,
    "naive Bayes": naive_bayes.make_model,
}
# Partial dependence: skip kNN (slow) and naive Bayes (poorly calibrated).
PD_MODELS = [m for m in MODELS if m not in ("kNN", "naive Bayes")]
PD_FEATURES = [
    "PRIOR_INVITATION_COUNT",
    "INVITATION_SEQUENCE",
    "MONTHS_SINCE_LAST_SCREEN",
    "GYN_PREVENTION_24M_CNT",
    "OTHER_SCREENING_36M_CNT",
    "HEALTHCARE_CONTACT_12M_CNT",
    "DENTAL_PREVENTION_24M_CNT",
    "MONTHS_LAST_INVITATION",
    "AGE_BAND",
    "GYN_REGISTERED_FLAG",
]


def clean(name: str) -> str:
    """'remainder__PRIOR_INVITATION_COUNT' -> 'PRIOR_INVITATION_COUNT'."""
    return name.split("__", 1)[-1]


def grid(values: pd.Series) -> list:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    values = values.dropna()
    if values.nunique() <= 20:
        return sorted(values.unique())
    return sorted(np.unique(np.round(np.quantile(values, np.linspace(0, 0.98, 20)))))


def partial_dependence(model: object, x: pd.DataFrame, feature: str) -> list[dict]:
    rows = x[x[feature].notna()] if x[feature].isna().any() else x
    out = []
    for value in grid(x[feature]):
        changed = rows.copy()
        if isinstance(x[feature].dtype, pd.CategoricalDtype):
            changed[feature] = pd.Categorical(
                [value] * len(changed), categories=x[feature].cat.categories
            )
        else:
            changed[feature] = value
        out.append(
            {
                "feature": feature,
                "value": str(value),
                "mean_prediction": model.predict_proba(changed)[:, 1].mean(),
                "rows": len(rows),
            }
        )
    return out


def draw_tree(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, y_test, depth: int):
    model = decision_tree.build(x, max_depth=depth, min_samples_leaf=500).fit(x, y)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    names = [clean(n) for n in model[0].get_feature_names_out()]
    width = {3: 22, 4: 34}[depth]
    fig, ax = plt.subplots(figsize=(width, 2.6 * depth + 2))
    plot_tree(
        model[-1],
        feature_names=names,
        class_names=["not screened", "screened"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=10 if depth == 3 else 8,
        ax=ax,
    )
    ax.set_title(
        f"Decision tree, depth {depth}, at least 500 rows per leaf "
        f"(test ROC AUC {auc:.3f}). 'value' = [share not screened, share screened]",
        fontsize=13,
    )
    fig.savefig(RESULTS / f"tree_depth{depth}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return model, auc


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data, x = prepare()
    y = {name: df[LABEL] for name, df in data.items()}
    sample = x["test"].sample(IMPORTANCE_ROWS, random_state=SEED).index
    pd_rows = x["train"].sample(PD_ROWS, random_state=SEED).index

    fitted, scores, importances, dependence = {}, [], [], []
    for name, make_model in MODELS.items():
        print(f"Fitting {name}...", flush=True)
        model = make_model(x["train"]).fit(x["train"], y["train"])
        fitted[name] = model
        auc = roc_auc_score(y["test"], model.predict_proba(x["test"])[:, 1])
        scores.append({"model": name, "test_roc_auc": auc})
        print(f"  test ROC AUC {auc:.4f}; permutation importance...", flush=True)
        result = permutation_importance(
            model,
            x["test"].loc[sample],
            y["test"].loc[sample],
            scoring="roc_auc",
            n_repeats=5,
            random_state=SEED,
        )
        importances += [
            {"model": name, "feature": f, "importance": m, "std": s}
            for f, m, s in zip(
                FEATURES, result.importances_mean, result.importances_std, strict=True
            )
        ]
        if name in PD_MODELS:
            print("  partial dependence...", flush=True)
            for feature in PD_FEATURES:
                rows = partial_dependence(model, x["train"].loc[pd_rows], feature)
                dependence += [{"model": name, **r} for r in rows]

    pd.DataFrame(scores).to_csv(RESULTS / "models.csv", index=False)
    pd.DataFrame(importances).to_csv(
        RESULTS / "permutation_importance.csv", index=False
    )
    pd.DataFrame(dependence).to_csv(RESULTS / "partial_dependence.csv", index=False)

    # Logistic regression: coefficients on scaled / one-hot inputs.
    lr = fitted["logistic regression"]
    coef = pd.DataFrame(
        {
            "input": [clean(n) for n in lr[0].get_feature_names_out()],
            "coefficient": lr[-1].coef_[0],
        }
    )
    coef["odds_ratio"] = np.exp(coef["coefficient"])
    coef.to_csv(RESULTS / "lr_coefficients.csv", index=False)

    # Trees: impurity importance, readable small trees, text rules of the real tree.
    tree_imp = []
    for name in ["decision tree", "random forest"]:
        model = fitted[name]
        names = [clean(n) for n in model[0].get_feature_names_out()]
        tree_imp += [
            {"model": name, "feature": f, "importance": v}
            for f, v in zip(names, model[-1].feature_importances_, strict=True)
        ]
    pd.DataFrame(tree_imp).to_csv(RESULTS / "tree_importance.csv", index=False)

    small = {}
    for depth in [3, 4]:
        _, auc = draw_tree(x["train"], y["train"], x["test"], y["test"], depth)
        small[depth] = auc
    tree = fitted["decision tree"]
    names = [clean(n) for n in tree[0].get_feature_names_out()]
    rules = export_text(tree[-1], feature_names=names, max_depth=4, show_weights=False)
    (RESULTS / "tree_rules.txt").write_text(
        "Real decision tree (depth 8, >=500 rows per leaf), top 4 levels; deeper "
        "branches are shown as 'truncated branch'. 'class: 1' = majority screened.\n\n"
        + rules,
        encoding="utf8",
    )
    pd.DataFrame(
        [
            {"model": f"decision tree depth {d}", "test_roc_auc": a}
            for d, a in small.items()
        ]
    ).to_csv(RESULTS / "models.csv", mode="a", header=False, index=False)

    encoder = tree[0].named_transformers_["categorical"]
    codes = [
        {"column": col, "code": i, "category": cat}
        for col, cats in zip(CATEGORICAL, encoder.categories_, strict=True)
        for i, cat in enumerate(cats)
    ]
    pd.DataFrame(codes).to_csv(RESULTS / "category_codes.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()

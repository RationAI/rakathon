"""Large ablation: do all model families converge to the same performance?

12M label, official patient-level split (fit on train, select on val, report test).
Three studies, every fit is one row of ``results/ablation.csv``:

- capacity: each model family from tiny to huge (and deliberately overfitting)
  settings, on all train rows and the 19 raw features;
- learning curve: four families trained on 1-100% of the train women;
- features: the 19 raw features vs the 96 engineered ones of the prepared data.

``results/best_per_family.csv``: per family the setting with the best val ROC AUC,
its test scores and a 95% bootstrap interval (resampling women) of its test ROC AUC
gap to the overall best model.

Usage (from training/classifiers, ~1 hour):
    uv run python -m ablation.run
"""

import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import CATEGORICAL, LABELS, SEED, ModelFactory, prepare
from decision_tree.model import build as tree
from gradient_boosting.model import build as boosting
from knn.model import build as knn
from logistic_regression.model import build as logistic
from mlp.model import build as mlp
from naive_bayes.model import build as naive_bayes
from random_forest.model import build as forest
from spline_logistic_regression.model import build as splines


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

LABEL = "SCREENING_12M_FLAG"
RESULTS = Path(__file__).resolve().parent / "results"
TRAIN_SAMPLE = 200_000  # rows used to measure the fit on the training data
KNN_ROWS = 50_000  # kNN prediction is O(train x test); fit it on a row sample
N_BOOTSTRAP = 200


# ---------------------------------------------------------------- ablation-only models


def few_features(_x: pd.DataFrame, columns: list[str]) -> object:
    """Logistic regression on a handful of columns (NaN -> 0 + missing indicator)."""
    numeric = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        StandardScaler(),
    )
    return make_pipeline(
        ColumnTransformer([("numeric", numeric, columns)]),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )


def constant(_x: pd.DataFrame) -> object:
    return DummyClassifier(strategy="prior")


# ---------------------------------------------------------------- configurations


@dataclass
class Config:
    family: str
    setting: str
    make: ModelFactory
    study: str = "capacity"
    train_fraction: float = 1.0  # share of train women
    max_rows: int | None = None  # row cap (kNN)
    features: str = "raw"


TOP = ["PRIOR_INVITATION_COUNT", "GYN_PREVENTION_24M_CNT", "MONTHS_SINCE_LAST_SCREEN"]
TOP5 = [*TOP, "FIRST_INVITATION", "OTHER_SCREENING_36M_CNT"]
OVERFIT_GB = {
    "max_leaf_nodes": 255,
    "min_samples_leaf": 1,
    "max_iter": 500,
    "early_stopping": False,
}

CAPACITY = [
    Config("constant", "base rate", constant),
    Config("few-feature LR", "1 feature", partial(few_features, columns=TOP[:1])),
    Config("few-feature LR", "3 features", partial(few_features, columns=TOP)),
    Config("few-feature LR", "5 features", partial(few_features, columns=TOP5)),
    Config("naive Bayes", "smoothing 1e-9", partial(naive_bayes, var_smoothing=1e-9)),
    Config("naive Bayes", "smoothing 1e-2", partial(naive_bayes, var_smoothing=1e-2)),
    *[
        Config("kNN (50k rows)", f"k={k}", partial(knn, k=k), max_rows=KNN_ROWS)
        for k in [5, 50, 250, 1000]
    ],
    *[
        Config("logistic regression", f"C={c:g}", partial(logistic, c=c))
        for c in [1e-4, 1e-2, 1, 100]
    ],
    Config("logistic regression", "splines (additive)", splines),
    *[
        Config("decision tree", f"depth {d}", partial(tree, max_depth=d))
        for d in [2, 4, 6, 8, 12, 16]
    ],
    Config(
        "decision tree",
        "depth 16, leaf 1",
        partial(tree, max_depth=16, min_samples_leaf=1),
    ),
    Config(
        "decision tree",
        "unlimited, leaf 1",
        partial(tree, max_depth=None, min_samples_leaf=1),
    ),
    Config(
        "random forest", "depth 8", partial(forest, max_depth=8, min_samples_leaf=1)
    ),
    Config(
        "random forest",
        "depth 16, leaf 20",
        partial(forest, max_depth=16, min_samples_leaf=20),
    ),
    Config(
        "random forest",
        "unlimited, leaf 1",
        partial(forest, max_depth=None, min_samples_leaf=1),
    ),
    Config("gradient boosting", "7 leaves", partial(boosting, max_leaf_nodes=7)),
    Config("gradient boosting", "31 leaves (default)", boosting),
    Config(
        "gradient boosting",
        "63 leaves, lr .05, 1000 iter",
        partial(boosting, max_leaf_nodes=63, learning_rate=0.05, max_iter=1000),
    ),
    Config(
        "gradient boosting",
        "255 leaves, lr .05, 1000 iter",
        partial(
            boosting,
            max_leaf_nodes=255,
            min_samples_leaf=20,
            learning_rate=0.05,
            max_iter=1000,
        ),
    ),
    Config(
        "gradient boosting",
        "255 leaves, 500 iter, no early stop",
        partial(boosting, **OVERFIT_GB),
    ),
    Config("MLP", "1 layer (64)", partial(mlp, hidden_layers=(64,))),
    Config("MLP", "2 layers (128-64)", partial(mlp, hidden_layers=(128, 64))),
    Config("MLP", "3 layers (256-128-64)", partial(mlp, hidden_layers=(256, 128, 64))),
    Config(
        "MLP",
        "4 layers (512-256-128-64)",
        partial(mlp, hidden_layers=(512, 256, 128, 64)),
    ),
    Config("MLP", "3 layers narrow (32-16-8)", partial(mlp, hidden_layers=(32, 16, 8))),
    Config(
        "MLP",
        "3 layers, strong L2 (alpha 1e-2)",
        partial(mlp, hidden_layers=(256, 128, 64), alpha=1e-2),
    ),
    Config(
        "MLP",
        "3 layers, 25 epochs, no early stop",
        partial(mlp, hidden_layers=(256, 128, 64), early_stopping=False, max_iter=25),
    ),
]

# Settings reused by the learning-curve and feature studies (same family + setting
# names as in CAPACITY, so the 100% / raw points come from the capacity study).
REFERENCE = {
    ("logistic regression", "C=1"): partial(logistic, c=1),
    ("decision tree", "depth 8"): partial(tree, max_depth=8),
    ("gradient boosting", "31 leaves (default)"): boosting,
    ("MLP", "2 layers (128-64)"): partial(mlp, hidden_layers=(128, 64)),
}
LEARNING_CURVE = [
    Config(family, setting, make, study="learning curve", train_fraction=fraction)
    for (family, setting), make in REFERENCE.items()
    for fraction in [0.01, 0.03, 0.1, 0.3]
]
FEATURES = [
    Config(family, setting, make, study="features", features="engineered")
    for (family, setting), make in REFERENCE.items()
    if family != "decision tree"
]
CONFIGS = CAPACITY + LEARNING_CURVE + FEATURES


# ---------------------------------------------------------------- running


def engineered_frames(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """All prepared columns except ids, the date string, labels and ``LIN_*``.

    ``LIN_*`` columns only re-encode other columns for linear models.
    """
    drop = {"INVITATION_ID", "PERSON_ID", "INVITATION_DATE", *LABELS}
    cols = [
        c for c in data["train"].columns if c not in drop and not c.startswith("LIN_")
    ]
    categories = {c: sorted(data["train"][c].dropna().unique()) for c in CATEGORICAL}
    frames = {}
    for name, df in data.items():
        frame = df[cols].copy()
        for c in CATEGORICAL:
            frame[c] = pd.Categorical(frame[c], categories=categories[c])
        frames[name] = frame
    return frames


def train_rows(
    config: Config, women_order: np.ndarray, train: pd.DataFrame
) -> pd.Index:
    rows = train.index
    if config.train_fraction < 1:
        chosen = women_order[: int(config.train_fraction * len(women_order))]
        rows = train.index[train["PERSON_ID"].isin(chosen)]
    if config.max_rows is not None and len(rows) > config.max_rows:
        rows = pd.Index(
            np.random.default_rng(SEED).choice(rows, config.max_rows, replace=False)
        )
    return rows


def proba(model: object, frame: pd.DataFrame, chunk: int = 50_000) -> np.ndarray:
    """Positive-class probability, predicted in chunks to bound memory (kNN)."""
    parts = [
        model.predict_proba(frame.iloc[i : i + chunk])[:, 1]
        for i in range(0, len(frame), chunk)
    ]
    return np.concatenate(parts)


def bootstrap_gap(
    y: np.ndarray, preds: dict[str, np.ndarray], best: str, women: pd.Series
) -> dict[str, tuple[float, float]]:
    """95% interval of AUC(model) - AUC(best), resampling women with all their rows."""
    codes, _ = pd.factorize(women)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(SEED)
    gaps = {name: [] for name in preds}
    for _ in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(counts), len(counts))
        n = counts[picked]
        idx = order[
            np.repeat(starts[picked], n)
            + np.arange(n.sum())
            - np.repeat(np.cumsum(n) - n, n)
        ]
        best_auc = roc_auc_score(y[idx], preds[best][idx])
        for name, p in preds.items():
            gaps[name].append(roc_auc_score(y[idx], p[idx]) - best_auc)
    return {name: tuple(np.percentile(g, [2.5, 97.5])) for name, g in gaps.items()}


def main() -> None:
    data, x_raw = prepare()
    frames = {"raw": x_raw, "engineered": engineered_frames(data)}
    y = {name: df[LABEL] for name, df in data.items()}
    women_order = (
        data["train"]["PERSON_ID"].drop_duplicates().sample(frac=1, random_state=SEED)
    ).to_numpy()

    rows, test_preds = [], {}
    for i, config in enumerate(CONFIGS, 1):
        x = frames[config.features]
        idx = train_rows(config, women_order, data["train"])
        print(
            f"[{i}/{len(CONFIGS)}] {config.study}: {config.family} - "
            f"{config.setting} ({len(idx):,} rows, {config.features})",
            flush=True,
        )
        start = time.time()
        model = config.make(x["train"].loc[idx]).fit(
            x["train"].loc[idx], y["train"][idx]
        )
        row = {
            "study": config.study,
            "family": config.family,
            "setting": config.setting,
            "features": config.features,
            "train_fraction": config.train_fraction,
            "train_rows": len(idx),
            "fit_seconds": round(time.time() - start),
        }
        for split in ["val", "test"]:
            p = proba(model, x[split])
            row[f"{split}_roc_auc"] = roc_auc_score(y[split], p)
            row[f"{split}_pr_auc"] = average_precision_score(y[split], p)
        if config.study == "capacity":
            test_preds[(config.family, config.setting)] = p
        sample = (
            idx
            if len(idx) <= TRAIN_SAMPLE
            else pd.Index(
                np.random.default_rng(SEED).choice(idx, TRAIN_SAMPLE, replace=False)
            )
        )
        p_train = proba(model, x["train"].loc[sample])
        row["train_roc_auc"] = roc_auc_score(y["train"][sample], p_train)
        rows.append(row)
        print(
            f"    val {row['val_roc_auc']:.4f}  test {row['test_roc_auc']:.4f}  "
            f"train {row['train_roc_auc']:.4f}  ({row['fit_seconds']}s)",
            flush=True,
        )

        RESULTS.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(
            RESULTS / "ablation.csv", index=False
        )  # keep progress

    results = pd.DataFrame(rows)
    capacity = results[results["study"] == "capacity"]
    best = capacity.loc[capacity.groupby("family", sort=False)["val_roc_auc"].idxmax()]
    overall = best.loc[best["val_roc_auc"].idxmax()]
    keys = {f"{r.family}": (r.family, r.setting) for r in best.itertuples()}
    gaps = bootstrap_gap(
        y["test"].to_numpy(),
        {family: test_preds[key] for family, key in keys.items()},
        overall["family"],
        data["test"]["PERSON_ID"],
    )
    best = best.assign(
        gap_to_best=best["test_roc_auc"] - overall["test_roc_auc"],
        gap_low=[gaps[f][0] for f in best["family"]],
        gap_high=[gaps[f][1] for f in best["family"]],
    ).sort_values("test_roc_auc", ascending=False)
    best.to_csv(RESULTS / "best_per_family.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(
            best[
                [
                    "family",
                    "setting",
                    "val_roc_auc",
                    "test_roc_auc",
                    "test_pr_auc",
                    "train_roc_auc",
                    "gap_to_best",
                    "gap_low",
                    "gap_high",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

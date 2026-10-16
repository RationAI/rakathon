"""Shared data loading and evaluation for the invitation-predictor baselines.

One row of the dataset is one invitation; for each invitation we predict whether the
woman is screened within 3, 6 and 12 months (``SCREENING_{3,6,12}M_FLAG``). Each
baseline fits one model per label on the train split and reports scores on the
validation and test splits written by ``preprocessing/split.py``.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
SEED = 0

LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
NOT_FEATURES = ["INVITATION_ID", "PERSON_ID", "INVITATION_DATE", *LABELS]
FLAGS = [
    "FIRST_INVITATION",
    "GP_REGISTERED",
    "GYN_REGISTERED_FLAG",
    "NEVER_SCREENED_BEFORE_FLAG",
]
CATEGORICAL = [
    "AGE_BAND",
    "VZP_TERRITORIAL_AREA",
    "GP_REG_DURATION_BAND",
    "GYN_REG_DURATION_BAND",
]


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(
        SPLITS / f"{name}.csv",
        dtype={"INVITATION_ID": str, "PERSON_ID": str},
        keep_default_na=False,
        na_values=[""],
    )
    for col in FLAGS + LABELS:
        df[col] = df[col].eq("A").astype(int)
    return df


def features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    # Missing values are structural (e.g. no GYN band when not registered) and are
    # left as NaN; each model decides how to handle them.
    X = df.drop(columns=NOT_FEATURES)
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def numeric_features(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if col not in CATEGORICAL + FLAGS]


def scores(y: pd.Series, p: pd.Series) -> dict[str, float]:
    return {
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "base rate": y.mean(),
    }


ModelFactory = Callable[[pd.DataFrame], ClassifierMixin]


def prepare() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load all splits; return (raw frames, feature frames) keyed by split name."""
    data = {name: load(name) for name in ["train", "val", "test"]}
    # Category sets come from train only; unseen values in val/test become NaN.
    categories = {
        col: sorted(data["train"][col].dropna().unique()) for col in CATEGORICAL
    }
    X = {name: features(df, categories) for name, df in data.items()}
    return data, X


def evaluate(
    make_model: ModelFactory,
    data: dict[str, pd.DataFrame],
    x: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Fit ``make_model(x_train)`` once per label; return val / test scores."""
    results = {}
    for label in LABELS:
        model = make_model(x["train"])
        model.fit(x["train"], data["train"][label])
        for name in ["val", "test"]:
            p = model.predict_proba(x[name])[:, 1]
            results[(label, name)] = scores(data[name][label], p)

    table = pd.DataFrame(results).T
    table.index.names = ["label", "split"]
    return table


def run(make_model: ModelFactory, title: str) -> None:
    data, X = prepare()
    print(title)
    print(f"Train rows: {len(data['train']):,} | features: {X['train'].shape[1]}")
    print(evaluate(make_model, data, X).to_string(float_format="{:.3f}".format))

"""Gradient-boosting baseline for the invitation predictor.

One row of the dataset is one invitation; for each invitation we predict whether the
woman is screened within 3, 6 and 12 months (``SCREENING_{3,6,12}M_FLAG``). This
baseline fits one untuned ``HistGradientBoostingClassifier`` per label on the train
split and reports scores on the validation and test splits.

Reads the files written by ``preprocessing/split.py``.

Usage:
    uv run python train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
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
    # Missing values are structural (e.g. no GYN band when not registered); the
    # trees handle NaN natively, so they stay as NaN.
    X = df.drop(columns=NOT_FEATURES)
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def scores(y: pd.Series, p: pd.Series) -> dict[str, float]:
    return {
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "base rate": y.mean(),
    }


def main() -> None:
    data = {name: load(name) for name in ["train", "val", "test"]}
    # Category sets come from train only; unseen values in val/test become NaN.
    categories = {
        col: sorted(data["train"][col].dropna().unique()) for col in CATEGORICAL
    }
    X = {name: features(df, categories) for name, df in data.items()}

    results = {}
    for label in LABELS:
        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", random_state=SEED
        )
        model.fit(X["train"], data["train"][label])
        for name in ["val", "test"]:
            p = model.predict_proba(X[name])[:, 1]
            results[(label, name)] = scores(data[name][label], p)

    table = pd.DataFrame(results).T
    table.index.names = ["label", "split"]
    print(f"Train rows: {len(data['train']):,} | features: {X['train'].shape[1]}")
    print(table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()

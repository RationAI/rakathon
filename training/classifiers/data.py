"""Shared data loading and evaluation for the invitation-predictor baselines.

One row of the dataset is one invitation; for each invitation we predict whether the
woman is screened within 3, 6 and 12 months (``SCREENING_{3,6,12}M_FLAG``). Each
baseline fits one model per label on the train split and reports scores on the
validation and test splits written by ``preprocessing/split.py``.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
SEED = 0

LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
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
NUMERIC = [
    "PRIOR_INVITATION_COUNT",
    "MONTHS_LAST_INVITATION",
    "INVITATION_SEQUENCE",
    "HEALTHCARE_CONTACT_12M_CNT",
    "INSURANCE_DURATION_YEARS",
    "PRIOR_SCREEN_COUNT_5Y",
    "MONTHS_SINCE_LAST_SCREEN",
    "GP_PREVENTION_24M_CNT",
    "GYN_PREVENTION_24M_CNT",
    "OTHER_SCREENING_36M_CNT",
    "DENTAL_PREVENTION_24M_CNT",
]
# The 19 original dataset columns. The prepared splits carry ~100 engineered columns
# as well; the baselines deliberately use only the raw ones.
FEATURES = NUMERIC + FLAGS + CATEGORICAL


def load(name: str) -> pd.DataFrame:
    # Prepared splits store flags and labels as 1/0.
    return pd.read_csv(
        SPLITS / f"{name}.csv",
        dtype={"INVITATION_ID": str, "PERSON_ID": str},
        keep_default_na=False,
        na_values=[""],
    )


def features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    # Missing values are structural (e.g. no GYN band when not registered) and are
    # left as NaN; each model decides how to handle them.
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def numeric_features(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if col not in CATEGORICAL + FLAGS]


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """log1p that keeps the sign; equals log1p for the (non-negative) raw counts."""
    return np.sign(values) * np.log1p(np.abs(values))


def to_dense(m: object) -> np.ndarray:
    """Sparse or dense matrix -> dense float32 (for models without sparse support)."""
    return (m.toarray() if hasattr(m, "toarray") else np.asarray(m)).astype(np.float32)


def scaled_one_hot(x: pd.DataFrame) -> ColumnTransformer:
    """Preprocessing for models that need numeric, scaled input (LR, MLP).

    - categorical columns: one-hot, missing ("not registered") as its own level,
      unseen levels ignored,
    - numeric NaNs (no previous invitation / screening): 0 plus a missing indicator,
    - numeric columns: signed log1p (counts are long-tailed), then standardised,
    - 0/1 flags: passed through.
    """
    numeric = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        FunctionTransformer(signed_log1p),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features(x)),
            ("categorical", categorical, CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ]
    )


def ordinal_encoded() -> ColumnTransformer:
    """Preprocessing for tree models: categorical columns as integer codes.

    Trees can still split the codes (less freely than one-hot); NaN stays NaN, which
    sklearn trees handle natively; all other columns pass through unchanged.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    return ColumnTransformer(
        [("categorical", encoder, CATEGORICAL)], remainder="passthrough"
    )


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

"""Global vs. segmented gradient boosting on the prepared splits.

The invitation population is a mix of very different groups (first-time invitees
respond ~50%, repeat invitees ~13%), so besides one global model per label we try
several ways of cutting the data into segments, fit one model per segment, and compare
pooled and per-segment scores against the global model on validation and test.

Input: ``data/splits/*.csv`` written by ``preprocessing/split.py`` (prepared dataset,
flags are 1/0). The ``LIN_*`` columns are linear-model copies and are not used here.

Usage:
    uv run python segmented.py
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
RESULTS = Path(__file__).with_name("segmented_results.csv")
SEED = 0

LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
CATEGORICAL = [
    "AGE_BAND",
    "VZP_TERRITORIAL_AREA",
    "GP_REG_DURATION_BAND",
    "GYN_REG_DURATION_BAND",
]
NOT_FEATURES = ["INVITATION_ID", "PERSON_ID", "INVITATION_DATE", *LABELS]


def labelled(df: pd.DataFrame, conditions: list[tuple[str, pd.Series]]) -> pd.Series:
    """Segment name per row; later conditions overwrite earlier ones."""
    seg = pd.Series(conditions[0][0], index=df.index)
    for name, mask in conditions[1:]:
        seg[mask] = name
    return seg


SEGMENTATIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "first invitation": lambda d: labelled(d, [
        ("repeat", d.FIRST_INVITATION == 0), ("first", d.FIRST_INVITATION == 1)]),
    "months since last invitation": lambda d: labelled(d, [
        ("<=13 months", d.MONTHS_LAST_INVITATION <= 13),
        (">13 months", d.MONTHS_LAST_INVITATION > 13),
        ("none (first)", d.MONTHS_LAST_INVITATION.isna())]),
    "healthcare contact 0": lambda d: labelled(d, [
        ("contact>0", d.HEALTHCARE_CONTACT_12M_CNT > 0),
        ("contact=0", d.HEALTHCARE_CONTACT_12M_CNT == 0)]),
    "insurance duration": lambda d: labelled(d, [
        ("<=20y", d.INSURANCE_DURATION_YEARS <= 20),
        (">20y", d.INSURANCE_DURATION_YEARS > 20)]),
    "GP registered": lambda d: labelled(d, [
        ("no GP", d.GP_REGISTERED == 0), ("GP", d.GP_REGISTERED == 1)]),
    "GYN registered": lambda d: labelled(d, [
        ("no GYN", d.GYN_REGISTERED_FLAG == 0), ("GYN", d.GYN_REGISTERED_FLAG == 1)]),
    "prior screen count": lambda d: labelled(d, [
        ("new (0)", d.PRIOR_SCREEN_COUNT_5Y == 0),
        ("regular (1-2)", d.PRIOR_SCREEN_COUNT_5Y.between(1, 2)),
        ("irregular (3+)", d.PRIOR_SCREEN_COUNT_5Y >= 3)]),
    "never screened": lambda d: labelled(d, [
        ("screened before", d.NEVER_SCREENED_BEFORE_FLAG == 0),
        ("never screened", d.NEVER_SCREENED_BEFORE_FLAG == 1)]),
    "first / never / screened before": lambda d: labelled(d, [
        ("repeat, screened before", d.FIRST_INVITATION == 0),
        ("repeat, never screened", d.NEVER_SCREENED_BEFORE_FLAG == 1),
        ("first", d.FIRST_INVITATION == 1)]),
}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(
        SPLITS / f"{name}.csv",
        dtype={"INVITATION_ID": str, "PERSON_ID": str},
        keep_default_na=False,
        na_values=[""],
    )


def features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    x = df.drop(columns=NOT_FEATURES)
    x = x[[c for c in x.columns if not c.startswith("LIN_")]]
    for col in CATEGORICAL:
        x[col] = pd.Categorical(x[col], categories=categories[col])
    return x


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=SEED,
    )


def scores(y: pd.Series, p: pd.Series) -> dict[str, float]:
    return {"ROC AUC": roc_auc_score(y, p), "PR AUC": average_precision_score(y, p)}


def fit_segmented(x: pd.DataFrame, y: pd.Series, seg: pd.Series) -> dict:
    models = {}
    for s in seg.unique():
        m = seg == s
        # constant columns inside a segment crash HGB, so drop them
        cols = x.columns[x[m].nunique() > 1]
        models[s] = (cols, make_model().fit(x.loc[m, cols], y[m]))
    return models


def predict_segmented(models: dict, x: pd.DataFrame, seg: pd.Series) -> pd.Series:
    p = pd.Series(np.nan, index=x.index)
    for s, (cols, model) in models.items():
        m = seg == s
        if m.any():
            p[m] = model.predict_proba(x.loc[m, cols])[:, 1]
    return p


def main() -> None:
    data = {name: load(name) for name in ["train", "val", "test"]}
    categories = {c: sorted(data["train"][c].dropna().unique()) for c in CATEGORICAL}
    x = {name: features(df, categories) for name, df in data.items()}
    print(f"features: {x['train'].shape[1]}")

    rows = []
    for label in LABELS:
        y_train = data["train"][label]
        glob = make_model().fit(x["train"], y_train)
        p_global = {
            s: pd.Series(glob.predict_proba(x[s])[:, 1], index=data[s].index)
            for s in ["val", "test"]
        }
        for seg_name, seg_fn in SEGMENTATIONS.items():
            print(f"{label} | {seg_name}", flush=True)
            seg = {name: seg_fn(df) for name, df in data.items()}
            models = fit_segmented(x["train"], y_train, seg["train"])
            for split in ["val", "test"]:
                y = data[split][label]
                p_seg = predict_segmented(models, x[split], seg[split])
                for s in ["ALL", *sorted(seg[split].unique())]:
                    m = slice(None) if s == "ALL" else (seg[split] == s)
                    g = scores(y[m], p_global[split][m])
                    sg = scores(y[m], p_seg[m])
                    rows.append({
                        "segmentation": seg_name, "label": label, "split": split,
                        "segment": s,
                        "n_train": len(y_train) if s == "ALL" else int((seg["train"] == s).sum()),
                        "rate": y[m].mean(),
                        "ROC global": g["ROC AUC"], "ROC segmented": sg["ROC AUC"],
                        "ROC delta": sg["ROC AUC"] - g["ROC AUC"],
                        "PR global": g["PR AUC"], "PR segmented": sg["PR AUC"],
                        "PR delta": sg["PR AUC"] - g["PR AUC"],
                    })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS, index=False)
    fmt = "{:.3f}".format
    print("\nPOOLED (all segments together), test split")
    pooled = table[(table.segment == "ALL") & (table.split == "test")]
    print(pooled.pivot(index="segmentation", columns="label", values="ROC delta")
          .to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\nPER SEGMENT, test split, 12M label")
    seg12 = table[(table.split == "test") & (table.label == "SCREENING_12M_FLAG")
                  & (table.segment != "ALL")]
    print(seg12[["segmentation", "segment", "n_train", "rate", "ROC global",
                 "ROC segmented", "ROC delta"]].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()

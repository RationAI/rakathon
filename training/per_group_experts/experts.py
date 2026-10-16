"""Per-group experts: does a model trained on one group predict it better?

For every grouping criterion (age, first invitation, ...) the train split is cut into
groups and one gradient-boosting "expert" is trained per group. Each expert is compared
with the global model (trained on all train rows) on the same group's test rows.

Within-group AUCs are lower than the overall AUC because the grouping variable itself
is a strong predictor; compare expert vs global inside a group, not across groups.

Writes to ``results/``:
- ``group_scores.csv``: per group n, base rate, global and expert ROC/PR AUC, expert
  gain with a 95% bootstrap interval (resampling women, not rows),
- ``group_importance.csv``: permutation importance of each expert on its test rows.

Usage:
    uv run python experts.py
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from data import CATEGORICAL, FEATURES, SEED, features, load


RESULTS = Path(__file__).resolve().parent / "results"
LABEL = "SCREENING_12M_FLAG"
N_BOOTSTRAP = 200
IMPORTANCE_ROWS = 20_000  # test rows per group used for permutation importance


def cut(col: str, bins: list[float], labels: list[str]) -> Callable:
    return lambda df: pd.cut(df[col], bins, labels=labels)


def flag(col: str, yes: str, no: str) -> Callable:
    return lambda df: pd.Categorical(
        np.where(df[col] == 1, yes, no), categories=[yes, no]
    )


AGE_GROUPS = {"40-44": "<50", "45-49": "<50", "65-69": "65+", "70-74": "65+"}


def area_regions() -> dict[str, str]:
    """Map each VZP area to its region (kraj), parsed from the geocoder's match.

    ``matched`` looks like "okres Benešov, Středočeský kraj, Česko"; Prague city is
    its own region and is matched as just "Praha, Česko". Built by
    ``features/spatial/02_geocode_areas.py``.
    """
    areas = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "area_coordinates.csv"
    )
    regions = {}
    for area, matched in zip(areas["area"], areas["matched"], strict=True):
        parts = [p.strip() for p in matched.split(",") if "kraj" in p.lower()]
        regions[area] = parts[0] if parts else "Praha"
    praha = [a for a, r in regions.items() if r == "Praha"]
    assert praha == ["PRAHA-HL.MĚSTO"], f"unexpected areas without a region: {praha}"
    return regions


def region(df: pd.DataFrame) -> pd.Categorical:
    regions = area_regions()
    mapped = df["VZP_TERRITORIAL_AREA"].map(regions)
    assert mapped.notna().all(), "VZP areas missing from area_coordinates.csv"
    return pd.Categorical(mapped, categories=sorted(set(regions.values())))


# criterion -> function(df) returning the group label of every row
CRITERIA: dict[str, Callable] = {
    "age": lambda df: pd.Categorical(
        df["AGE_BAND"].map(AGE_GROUPS).fillna("50-64"),
        categories=["<50", "50-64", "65+"],
    ),
    "first invitation": flag("FIRST_INVITATION", "first", "repeat"),
    "screening history": flag(
        "NEVER_SCREENED_BEFORE_FLAG", "never screened (5y)", "screened before"
    ),
    "invitation sequence": cut(
        "INVITATION_SEQUENCE", [0, 1, 3, 6, np.inf], ["1", "2-3", "4-6", "7+"]
    ),
    "gynaecologist": flag("GYN_REGISTERED_FLAG", "registered", "not registered"),
    "healthcare contact (12m)": cut(
        "HEALTHCARE_CONTACT_12M_CNT", [-1, 0, 10, np.inf], ["0", "1-10", "11+"]
    ),
    "distance to centre": cut(
        "SPATIAL_DIST_NEAREST_KM",
        [-1, 10, 25, np.inf],
        ["<=10 km", "10-25 km", ">25 km"],
    ),
    # 75 VZP areas are too small for one expert each; group them into 14 regions.
    "region": region,
}


def fit(x: pd.DataFrame, y: pd.Series) -> tuple[HistGradientBoostingClassifier, list]:
    # Within a group some features are constant (e.g. FIRST_INVITATION in "first");
    # HistGradientBoosting can fail on constant columns, so drop them.
    cols = [c for c in x.columns if x[c].nunique(dropna=False) > 1]
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=SEED
    )
    return model.fit(x[cols], y), cols


def bootstrap_gain(
    y: np.ndarray, p_expert: np.ndarray, p_global: np.ndarray, women: pd.Series
) -> tuple[float, float]:
    """95% interval of AUC(expert) - AUC(global), resampling women with all their rows."""
    codes, _ = pd.factorize(women)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(counts), len(counts))
        n = counts[picked]
        offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        idx = order[np.repeat(starts[picked], n) + offsets]
        gains.append(
            roc_auc_score(y[idx], p_expert[idx]) - roc_auc_score(y[idx], p_global[idx])
        )
    low, high = np.percentile(gains, [2.5, 97.5])
    return low, high


def importance(
    model: HistGradientBoostingClassifier, cols: list, x: pd.DataFrame, y: pd.Series
) -> pd.Series:
    sample = x.sample(min(len(x), IMPORTANCE_ROWS), random_state=SEED).index
    result = permutation_importance(
        model,
        x.loc[sample, cols],
        y.loc[sample],
        scoring="roc_auc",
        n_repeats=3,
        random_state=SEED,
        n_jobs=-1,
    )
    # Features dropped as constant get 0 importance.
    return pd.Series(result.importances_mean, index=cols).reindex(
        FEATURES, fill_value=0
    )


def main() -> None:
    train, test = load("train"), load("test")
    categories = {col: sorted(train[col].dropna().unique()) for col in CATEGORICAL}
    x_train, x_test = features(train, categories), features(test, categories)
    y_train, y_test = train[LABEL], test[LABEL]

    print("Fitting global model...", flush=True)
    global_model, global_cols = fit(x_train, y_train)
    p_global = pd.Series(
        global_model.predict_proba(x_test[global_cols])[:, 1], index=test.index
    )

    scores = [
        {
            "criterion": "all",
            "group": "all",
            "n_train": len(train),
            "n_test": len(test),
            "base_rate": y_test.mean(),
            "auc_global": roc_auc_score(y_test, p_global),
            "auc_expert": roc_auc_score(y_test, p_global),
            "pr_auc_global": average_precision_score(y_test, p_global),
            "pr_auc_expert": average_precision_score(y_test, p_global),
            "gain_low": 0.0,
            "gain_high": 0.0,
        }
    ]
    importances = [
        importance(global_model, global_cols, x_test, y_test).rename(("all", "all"))
    ]

    for criterion, grouping in CRITERIA.items():
        g_train, g_test = grouping(train), grouping(test)
        for group in pd.Categorical(g_test).categories:
            tr, te = np.asarray(g_train == group), np.asarray(g_test == group)
            print(f"{criterion} = {group}: {tr.sum():,} train rows", flush=True)
            model, cols = fit(x_train[tr], y_train[tr])
            y, pg = y_test[te], p_global[te]
            pe = pd.Series(
                model.predict_proba(x_test.loc[te, cols])[:, 1], index=y.index
            )
            low, high = bootstrap_gain(
                y.to_numpy(), pe.to_numpy(), pg.to_numpy(), test.loc[te, "PERSON_ID"]
            )
            scores.append(
                {
                    "criterion": criterion,
                    "group": group,
                    "n_train": int(tr.sum()),
                    "n_test": int(te.sum()),
                    "base_rate": y.mean(),
                    "auc_global": roc_auc_score(y, pg),
                    "auc_expert": roc_auc_score(y, pe),
                    "pr_auc_global": average_precision_score(y, pg),
                    "pr_auc_expert": average_precision_score(y, pe),
                    "gain_low": low,
                    "gain_high": high,
                }
            )
            importances.append(
                importance(model, cols, x_test[te], y).rename((criterion, group))
            )

    RESULTS.mkdir(exist_ok=True)
    scores = pd.DataFrame(scores)
    scores["gain"] = scores["auc_expert"] - scores["auc_global"]
    scores.to_csv(RESULTS / "group_scores.csv", index=False)

    imp = pd.DataFrame(importances)
    imp.index = pd.MultiIndex.from_tuples(imp.index, names=["criterion", "group"])
    imp.to_csv(RESULTS / "group_importance.csv")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(scores.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

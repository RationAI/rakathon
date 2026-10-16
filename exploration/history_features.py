"""History features built from earlier invitations of the same woman, and baseline comparison.

Leakage rule: an earlier invitation's outcome flag of horizon h is used for row i only if that flag's
window has already closed at T0 of row i (T0_i - T0_prev > h days).

Run: uv run python history_features.py [path/to/Rakhaton2026.csv]
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/Rakhaton2026.csv"
LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
FLAGS = ["GP_REGISTERED", "GYN_REGISTERED_FLAG", "FIRST_INVITATION", "NEVER_SCREENED_BEFORE_FLAG", *LABELS]
NUM = [
    "PRIOR_INVITATION_COUNT", "MONTHS_LAST_INVITATION", "INVITATION_SEQUENCE", "HEALTHCARE_CONTACT_12M_CNT",
    "INSURANCE_DURATION_YEARS", "PRIOR_SCREEN_COUNT_5Y", "MONTHS_SINCE_LAST_SCREEN", "GP_PREVENTION_24M_CNT",
    "GYN_PREVENTION_24M_CNT", "OTHER_SCREENING_36M_CNT", "DENTAL_PREVENTION_24M_CNT",
]
CAT = ["AGE_BAND", "VZP_TERRITORIAL_AREA", "GP_REG_DURATION_BAND", "GYN_REG_DURATION_BAND"]
MAX_LAG = 4  # a woman has at most 5 rows


def load() -> pd.DataFrame:
    df = pd.read_csv(PATH, parse_dates=["INVITATION_DATE"], date_format="%d.%m.%Y", keep_default_na=False,
                     na_values=[""])
    for c in FLAGS:
        df[c] = (df[c] == "A").astype(np.int8)
    for c in CAT:
        df[c] = df[c].fillna("none").astype("category")
    return df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"]).reset_index(drop=True)


def add_history(df: pd.DataFrame) -> list[str]:
    g = df.groupby("PERSON_ID", sort=False)
    t0 = df.INVITATION_DATE.values.astype("datetime64[D]").astype(np.int64)
    n = len(df)
    n_known = np.zeros(n, np.int16)
    n_resp6 = np.zeros(n, np.int16)
    n_resp3 = np.zeros(n, np.int16)
    n_resp12 = np.zeros(n, np.int16)
    streak = np.zeros(n, np.int16)
    streak_alive = np.ones(n, bool)
    last_known = np.full(n, np.nan)
    last_resp6 = np.full(n, np.nan)
    last_resp3 = np.full(n, np.nan)
    gap_prev = np.full(n, np.nan)
    d_contact = np.full(n, np.nan)
    d_gynprev = np.full(n, np.nan)
    d_gp = np.full(n, np.nan)
    d_gyn = np.full(n, np.nan)
    first_t0 = g.INVITATION_DATE.transform("min").values.astype("datetime64[D]").astype(np.int64)

    pid = df.PERSON_ID.values
    for lag in range(1, MAX_LAG + 1):
        valid = g.PERSON_ID.shift(lag).notna().values
        sh = lambda c: g[c].shift(lag).values  # noqa: E731
        gap = t0 - np.nan_to_num(g.INVITATION_DATE.shift(lag).values.astype("datetime64[D]").astype(np.float64)
                                 ).astype(np.int64)
        # NaT shifts become huge negatives after astype; guard with valid
        gap = np.where(valid, gap, 0)
        known6 = valid & (gap > 180)
        known3 = valid & (gap > 90)
        known12 = valid & (gap > 365)
        f3, f6, f12 = (np.nan_to_num(sh(c)) for c in LABELS)
        n_known += known6
        n_resp6 += known6 & (f6 == 1)
        n_resp3 += known3 & (f3 == 1)
        n_resp12 += known12 & (f12 == 1)
        if lag == 1:
            last_known = np.where(known6, 1.0, np.nan)
            last_resp6 = np.where(known6, f6, np.nan)
            last_resp3 = np.where(known3, f3, np.nan)
            gap_prev = np.where(valid, gap, np.nan)
            d_contact = np.where(valid, df.HEALTHCARE_CONTACT_12M_CNT.values - sh("HEALTHCARE_CONTACT_12M_CNT"), np.nan)
            d_gynprev = np.where(valid, df.GYN_PREVENTION_24M_CNT.values - sh("GYN_PREVENTION_24M_CNT"), np.nan)
            d_gp = np.where(valid, df.GP_REGISTERED.values - sh("GP_REGISTERED"), np.nan)
            d_gyn = np.where(valid, df.GYN_REGISTERED_FLAG.values - sh("GYN_REGISTERED_FLAG"), np.nan)
        # consecutive known non-responses, counted from the most recent invitation backwards
        streak_alive &= known6 & (f6 == 0)
        streak += streak_alive

    df["H_N_PREV_KNOWN"] = n_known
    df["H_N_RESP_6M"] = n_resp6
    df["H_N_RESP_3M"] = n_resp3
    df["H_N_RESP_12M"] = n_resp12
    df["H_RESP_RATE_6M"] = np.where(n_known > 0, n_resp6 / np.maximum(n_known, 1), np.nan)
    df["H_FAST_SHARE"] = np.where(n_resp6 > 0, n_resp3 / np.maximum(n_resp6, 1), np.nan)
    df["H_NONRESP_STREAK"] = streak
    df["H_LAST_RESP_6M"] = last_resp6
    df["H_LAST_RESP_3M"] = last_resp3
    df["H_GAP_PREV_DAYS"] = gap_prev
    df["H_DAYS_SINCE_FIRST"] = t0 - first_t0
    df["H_D_CONTACT"] = d_contact
    df["H_D_GYN_PREV"] = d_gynprev
    df["H_D_GP_REG"] = d_gp
    df["H_D_GYN_REG"] = d_gyn
    return [c for c in df.columns if c.startswith("H_")]


def leakage_check(df: pd.DataFrame) -> None:
    """Recompute a few women's features by brute force and compare."""
    rng = np.random.default_rng(0)
    multi = df[df.INVITATION_SEQUENCE > 1].PERSON_ID.unique()
    for pid in rng.choice(multi, 300, replace=False):
        w = df[df.PERSON_ID == pid].reset_index(drop=True)
        for i in range(len(w)):
            prev = w.iloc[:i]
            known = prev[(w.INVITATION_DATE[i] - prev.INVITATION_DATE).dt.days > 180]
            assert w.H_N_PREV_KNOWN[i] == len(known), pid
            assert w.H_N_RESP_6M[i] == known.SCREENING_6M_FLAG.sum(), pid
            s = 0
            for _, r in known.iloc[::-1].iterrows():
                if r.SCREENING_6M_FLAG:
                    break
                s += 1
            # streak breaks at an unknown (too recent) invitation, so it can be lower than brute force
            assert w.H_NONRESP_STREAK[i] <= s, pid
    print("leakage check passed")


def evaluate(df: pd.DataFrame, feats: list[str], name: str, split: str) -> pd.DataFrame:
    year = df.INVITATION_DATE.dt.year
    # 12M windows of the last train year must close before the first test invitation
    train, test = (year <= 2021, year >= 2023) if split == "A:20-21>23-24" else (year <= 2022, year == 2024)
    feats = [f for f in feats if df.loc[train, f].nunique() > 1]  # constant columns break HGB binning
    rows = []
    for lab in LABELS:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, categorical_features="from_dtype",
                                             early_stopping=True, random_state=0)
        clf.fit(df.loc[train, feats], df.loc[train, lab])
        p = clf.predict_proba(df.loc[test, feats])[:, 1]
        y = df.loc[test, lab]
        first = df.loc[test, "FIRST_INVITATION"] == 1
        k = int(0.1 * len(y))
        top = y.values[np.argsort(-p)[:k]].mean()
        rows.append({
            "split": split, "model": name, "label": lab, "ROC": roc_auc_score(y, p), "PR": average_precision_score(y, p),
            "ROC_first": roc_auc_score(y[first], p[first]), "ROC_repeat": roc_auc_score(y[~first], p[~first]),
            "PR_repeat": average_precision_score(y[~first], p[~first]), "top10%_rate": top, "base": y.mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    hist = add_history(df)
    leakage_check(df)
    base = [*NUM, *CAT, "GP_REGISTERED", "GYN_REGISTERED_FLAG", "FIRST_INVITATION", "NEVER_SCREENED_BEFORE_FLAG"]
    res = pd.concat([evaluate(df, f, n, sp) for sp in ["A:20-21>23-24", "B:20-22>24"]
                     for n, f in [("baseline", base), ("baseline+history", base + hist)]])
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    res.to_csv("history_comparison.csv", index=False)

"""Clean Rakhaton2026.csv and add engineered features. Writes a new CSV; the raw file is not modified.

Run: uv run python prepare_data.py [in.csv] [out.csv]
"""

import re
import sys

import numpy as np
import pandas as pd

from history_features import add_history  # noqa: E402

SPATIAL = "../data/area_spatial_features.csv"  # built by ../spatial/01-03 scripts
SRC = sys.argv[1] if len(sys.argv) > 1 else "../data/Rakhaton2026.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "../data/Rakhaton2026_prepared.csv"

YES_NO = [
    "FIRST_INVITATION", "GP_REGISTERED", "GYN_REGISTERED_FLAG", "NEVER_SCREENED_BEFORE_FLAG",
    "SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG",
]
DURATION_ORDINAL = {"": 0, "<2": 1, "2-5": 2, ">5": 3}  # 0 = not registered / unknown


def age_band_mean(band: str) -> float:
    """Mean age of the group: '45-49' -> 47, '<25' -> 22 (assumes 20-24), '75+' -> 77 (assumes 75-79)."""
    if m := re.fullmatch(r"(\d+)-(\d+)", band):
        return (int(m[1]) + int(m[2])) / 2
    if m := re.fullmatch(r"<(\d+)", band):
        return int(m[1]) - 3
    if m := re.fullmatch(r"(\d+)\+", band):
        return int(m[1]) + 2
    raise ValueError(f"unexpected AGE_BAND {band!r}")


def fix_sequence_starting_at_2(df: pd.DataFrame) -> int:
    """Women whose first row is FIRST_INVITATION=A but INVITATION_SEQUENCE=2: shift their sequence down by 1."""
    first = df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"]).groupby("PERSON_ID").head(1)
    bad = first[(first.FIRST_INVITATION == "A") & (first.INVITATION_SEQUENCE == 2)].PERSON_ID
    mask = df.PERSON_ID.isin(bad)
    df.loc[mask, "INVITATION_SEQUENCE"] -= 1
    return bad.nunique()


def main() -> None:
    df = pd.read_csv(SRC, dtype={"PERSON_ID": str, "INVITATION_ID": str}, keep_default_na=False, na_values=[""])
    n_rows = len(df)
    df["INVITATION_DATE"] = pd.to_datetime(df.INVITATION_DATE, format="%d.%m.%Y")

    # 1. fix sequences that start at 2 for first invitations
    n_fixed = fix_sequence_starting_at_2(df)
    print(f"sequence shifted for {n_fixed} women")

    # 2. date features
    d = df.INVITATION_DATE
    df["INVITATION_YEAR"] = d.dt.year
    df["INVITATION_MONTH"] = d.dt.month
    df["INVITATION_QUARTER"] = d.dt.quarter
    df["INVITATION_DAYS_SINCE_2020"] = (d - pd.Timestamp("2020-01-01")).dt.days

    # 3. invitations counted in PRIOR_INVITATION_COUNT but not in the current sequence run
    df["INVITATION_COUNT_DIFF"] = df.PRIOR_INVITATION_COUNT - df.INVITATION_SEQUENCE

    # 4. age band -> mean age of the group
    df["AGE_BAND_MEAN"] = df.AGE_BAND.map(age_band_mean)

    # 5. A/N -> 1/0
    for c in YES_NO:
        df[c] = (df[c] == "A").astype(np.int8)

    # 6. registration duration bands -> ordinal (0 = not registered)
    for c in ["GP_REG_DURATION_BAND", "GYN_REG_DURATION_BAND"]:
        df[c + "_ORD"] = df[c].fillna("").map(DURATION_ORDINAL).astype(np.int8)
    df["INSURANCE_DURATION_BAND_ORD"] = pd.cut(df.INSURANCE_DURATION_YEARS, [-1, 1, 5, 1000], labels=[1, 2, 3]).astype(
        np.int8)

    # 7. sequence restarted after a long break: sequence does not grow vs. the woman's previous row
    ordered = df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"])
    prev_seq = ordered.groupby("PERSON_ID").INVITATION_SEQUENCE.shift()
    df["SEQUENCE_RESET_FLAG"] = (df.INVITATION_SEQUENCE <= prev_seq.reindex(df.index)).astype(np.int8)

    # 8. screening recency
    df["SCREENED_LAST_3M_FLAG"] = (df.MONTHS_SINCE_LAST_SCREEN <= 3).astype(np.int8)
    df["SCREEN_OVERDUE_30M_FLAG"] = (df.MONTHS_SINCE_LAST_SCREEN >= 30).astype(np.int8)

    # 9. healthcare contact and prevention
    df["HEALTHCARE_CONTACT_LOG"] = np.log1p(df.HEALTHCARE_CONTACT_12M_CNT)
    df["HEALTHCARE_CONTACT_ZERO_FLAG"] = (df.HEALTHCARE_CONTACT_12M_CNT == 0).astype(np.int8)
    prev = df[["GP_PREVENTION_24M_CNT", "GYN_PREVENTION_24M_CNT", "OTHER_SCREENING_36M_CNT",
               "DENTAL_PREVENTION_24M_CNT"]]
    df["PREVENTION_TOTAL"] = prev.sum(axis=1)
    df["PREVENTION_TYPES"] = (prev > 0).sum(axis=1).astype(np.int8)

    # 10. history of the same woman (earlier rows only, outcomes only when their window had closed)
    df = df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"]).reset_index(drop=True)
    add_history(df)
    df["H_HAS_PREV_ROW_FLAG"] = df.H_GAP_PREV_DAYS.notna().astype(np.int8)
    df["H_HAS_KNOWN_OUTCOME_FLAG"] = (df.H_N_PREV_KNOWN > 0).astype(np.int8)

    # 11. spatial features per VZP_TERRITORIAL_AREA (see ../spatial/README.md)
    spatial = pd.read_csv(SPATIAL)
    df = df.merge(spatial, on="VZP_TERRITORIAL_AREA", how="left", validate="many_to_one")
    assert df.SPATIAL_DIST_NEAREST_KM.notna().all(), "areas without spatial features"
    df["SPATIAL_NO_WAIT_30KM_FLAG"] = df.SPATIAL_WAIT_MIN_WITHIN_30KM_DAYS.isna().astype(np.int8)
    df["SPATIAL_NO_HOURS_30KM_FLAG"] = df.SPATIAL_EARLIEST_OPEN_30KM_H.isna().astype(np.int8)

    # 12. flags for linear models (LIN_*): dummies, bucket and "any" flags, median-filled copies of NaN columns
    for band in sorted(df.AGE_BAND.unique()):
        df["LIN_AGE_" + band.replace("-", "_").replace("<", "LT").replace("+", "P")] = (df.AGE_BAND == band).astype(
            np.int8)
    seq = df.INVITATION_SEQUENCE
    for name, cond in [("1", seq == 1), ("2", seq == 2), ("3_4", seq.between(3, 4)), ("5_6", seq.between(5, 6)),
                       ("7P", seq >= 7)]:
        df["LIN_SEQ_" + name] = cond.astype(np.int8)
    df["LIN_PRIOR_SCREEN_ANY"] = (df.PRIOR_SCREEN_COUNT_5Y >= 1).astype(np.int8)
    df["LIN_PRIOR_SCREEN_2P"] = (df.PRIOR_SCREEN_COUNT_5Y >= 2).astype(np.int8)  # regular 2-year screener
    for src, name in [("GP_PREVENTION_24M_CNT", "GP_PREV"), ("GYN_PREVENTION_24M_CNT", "GYN_PREV"),
                      ("OTHER_SCREENING_36M_CNT", "OTHER_SCREEN"), ("DENTAL_PREVENTION_24M_CNT", "DENTAL_PREV")]:
        df[f"LIN_{name}_ANY"] = (df[src] > 0).astype(np.int8)
    nan_cols = [c for c in df.columns if df[c].dtype.kind == "f" and df[c].isna().any()]
    for c in nan_cols:
        df["LIN_" + c] = df[c].fillna(df[c].median())  # use together with the matching *_FLAG / missing indicator
    print("median-filled for linear models:", nan_cols)

    assert len(df) == n_rows
    assert not ((df.FIRST_INVITATION == 1) & (df.INVITATION_SEQUENCE != 1)).any()
    assert df.INVITATION_COUNT_DIFF.min() >= 0
    df["INVITATION_DATE"] = d.dt.strftime("%Y-%m-%d")
    df.to_csv(DST, index=False)
    print(f"wrote {DST}: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()

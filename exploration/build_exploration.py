"""Build exploration/exploration.html: the whole exploration as one interactive page.

Same approach and style as build_report.py / build_web_map.py: every number is computed
here from the prepared data and embedded into exploration.template.html, so the page is
self-contained. Covers the labels, time and COVID, invitation history, response by
attribute, correlations with the labels, regions and the data-quality notes.

Run: uv run python build_exploration.py   (needs data/Rakhaton2026_prepared.csv)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE.parent / "segmentation"))
from segments import assign


LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
P = ["p3", "p6", "p12"]
CATEGORICAL = [
    "AGE_BAND",
    "VZP_TERRITORIAL_AREA",
    "GP_REG_DURATION_BAND",
    "GYN_REG_DURATION_BAND",
    "KRAJ",
]
ORIGINAL = set(pd.read_csv(DATA / "Rakhaton2026.csv", nrows=0).columns)

df = pd.read_csv(
    DATA / "Rakhaton2026_prepared.csv",
    dtype={"PERSON_ID": str, "INVITATION_ID": str},
    keep_default_na=False,
    na_values=[""],
).copy()  # consolidate the wide frame before adding columns
date = pd.to_datetime(df.INVITATION_DATE)
df = df.assign(  # all at once: adding columns one by one fragments the wide frame
    **{p: df[label] for p, label in zip(P, LABELS, strict=True)},
    month=date.dt.to_period("M").astype(str),
    year=date.dt.year,
)
print(f"{len(df):,} invitations loaded", flush=True)


def counts(g: pd.DataFrame) -> dict:
    return {"n": len(g), "p": [int(g[c].sum()) for c in P]}


def breakdown(
    key: pd.Series, order: list | None = None, names: dict | None = None
) -> list:
    rows = []
    for k, g in df.groupby(key, observed=True, dropna=False, sort=True):
        name = "missing" if pd.isna(k) else (names or {}).get(k, str(k))
        rows.append({"k": name, **counts(g)})
    if order:
        rows.sort(key=lambda r: order.index(r["k"]) if r["k"] in order else len(order))
    return rows


def capped(col: str, cap: int) -> pd.Series:
    s = df[col].clip(upper=cap).astype("Int64")
    return s.map(lambda v: f"{v}+" if v == cap else str(v))


def binned(col: str, bins: list, labels: list, missing: str | None = None) -> pd.Series:
    s = pd.cut(df[col], bins, labels=labels).astype(object)
    return s.where(df[col].notna(), missing) if missing else s


# ---------------------------------------------------------------- overview & labels
meta = {
    "rows": len(df),
    "women": int(df.PERSON_ID.nunique()),
    "from": str(date.min().date()),
    "to": str(date.max().date()),
    "rates": [float(df[c].mean()) for c in P],
    "first": float(df.FIRST_INVITATION.mean()),
    "areas": int(df.VZP_TERRITORIAL_AREA.nunique()),
    "dates": int(date.nunique()),
    "repeat_rows": float(df.PERSON_ID.map(df.PERSON_ID.value_counts()).gt(1).mean()),
}

# ---------------------------------------------------------------- time & COVID
months = pd.period_range(date.min(), date.max(), freq="M").astype(str)
by_month = df.groupby("month")
monthly = [
    {
        "m": m,
        **(
            counts(by_month.get_group(m))
            if m in by_month.groups
            else {"n": 0, "p": [0, 0, 0]}
        ),
    }
    for m in months
]
yearly = [{"k": str(y), **counts(g)} for y, g in df.groupby("year")]
seg = assign(df)
later = df.year >= 2021
rate_later = df.loc[later, "p12"].groupby(seg[later], observed=False).mean()
y20 = df.year == 2020
covid = {
    "expected_2020": float(seg[y20].map(rate_later).astype(float).mean()),
    "actual_2020": float(df.loc[y20, "p12"].mean()),
    "segments": [
        {
            "k": s.split(" ", 1)[1],
            "letter": s[0],
            "years": {
                str(y): float(r) for y, r in g.groupby(df.year)["p12"].mean().items()
            },
        }
        for s, g in df.groupby(seg, observed=False)
    ],
}

# ---------------------------------------------------------------- invitation history
age_order = (
    df.drop_duplicates("AGE_BAND").sort_values("AGE_BAND_MEAN").AGE_BAND.tolist()
)
mls_bins = [-1, 3, 29, 36, 42, 48, 54, 60, 100]
mls_labels = ["0-3", "4-29", "30-36", "37-42", "43-48", "49-54", "55-60", "61+"]
history = {
    "sequence": breakdown(df.INVITATION_SEQUENCE.astype(int)),
    "since_screen": breakdown(
        binned("MONTHS_SINCE_LAST_SCREEN", mls_bins, mls_labels, "never (5 y)"),
        order=[*mls_labels, "never (5 y)"],
    ),
}

# ---------------------------------------------------------------- response by attribute
YESNO = {1: "yes", 0: "no"}
band_order = ["<2", "2-5", ">5", "missing"]
attributes = {
    "Age band": breakdown(df.AGE_BAND, order=age_order),
    "First invitation": breakdown(df.FIRST_INVITATION, names=YESNO),
    "Invitation sequence": history["sequence"],
    "Months since last screen": history["since_screen"],
    "Screens in last 5 years": breakdown(
        capped("PRIOR_SCREEN_COUNT_5Y", 4), order=["0", "1", "2", "3", "4+"]
    ),
    "Gynaecologist prevention (2 y)": breakdown(
        capped("GYN_PREVENTION_24M_CNT", 3), order=["0", "1", "2", "3+"]
    ),
    "GP prevention (2 y)": breakdown(
        capped("GP_PREVENTION_24M_CNT", 2), order=["0", "1", "2+"]
    ),
    "Other screening (3 y)": breakdown(
        capped("OTHER_SCREENING_36M_CNT", 4), order=["0", "1", "2", "3", "4+"]
    ),
    "Dental prevention (2 y)": breakdown(
        capped("DENTAL_PREVENTION_24M_CNT", 3), order=["0", "1", "2", "3+"]
    ),
    "Healthcare contacts (12 m)": breakdown(
        binned(
            "HEALTHCARE_CONTACT_12M_CNT",
            [-1, 0, 2, 5, 10, 20, 50, 1000],
            ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"],
        ),
        order=["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"],
    ),
    "Gynaecologist registration": breakdown(df.GYN_REG_DURATION_BAND, order=band_order),
    "GP registration": breakdown(df.GP_REG_DURATION_BAND, order=band_order),
    "Insurance (years)": breakdown(
        binned(
            "INSURANCE_DURATION_YEARS",
            [0, 10, 20, 25, 30, 40],
            ["5-10", "11-20", "21-25", "26-30", "31+"],
        ),
        order=["5-10", "11-20", "21-25", "26-30", "31+"],
    ),
    "Region": breakdown(df.KRAJ),
}
for k in ["Gynaecologist registration", "GP registration"]:
    for r in attributes[k]:
        if r["k"] == "missing":
            r["k"] = "not registered"

# ---------------------------------------------------------------- correlations with the labels
numeric = [
    c
    for c in df.columns
    if c
    not in {
        "INVITATION_ID",
        "PERSON_ID",
        "INVITATION_DATE",
        "month",
        "year",
        *LABELS,
        *P,
        *CATEGORICAL,
    }
    and not c.startswith("LIN_")
    and pd.api.types.is_numeric_dtype(df[c])
]
print(f"Spearman correlations for {len(numeric)} attributes...", flush=True)
corr = []
for col in numeric:
    present = df[col].notna()
    x = df.loc[present, col].rank()
    row = {
        "k": col,
        "coverage": float(present.mean()),
        "origin": "history"
        if col.startswith("H_")
        else "spatial"
        if col.startswith("SPATIAL_")
        else "original"
        if col in ORIGINAL
        else "engineered",
        "rho": [
            float(np.corrcoef(x, df.loc[present, lab].rank())[0, 1]) for lab in LABELS
        ],
    }
    corr.append(row)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y).to_numpy().astype(float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


cramers = [
    {
        "k": c,
        "v": [cramers_v(df[c].fillna("not registered"), df[lab]) for lab in LABELS],
        "levels": int(df[c].nunique()),
    }
    for c in CATEGORICAL
]

data = {
    "meta": meta,
    "monthly": monthly,
    "yearly": yearly,
    "covid": covid,
    "history": history,
    "attributes": attributes,
    "corr": corr,
    "cramers": cramers,
}
html = (HERE / "exploration.template.html").read_text(encoding="utf-8")
out = HERE / "exploration.html"
out.write_text(
    html.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    ),
    encoding="utf-8",
)
print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")

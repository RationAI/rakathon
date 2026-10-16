"""Explainable segments of invitations, for assigning interventions.

Every invitation gets exactly one segment, by simple rules on information known at the
invitation date (never the outcome). The rules follow the three questions that drive
response in the models (see ``training/classifiers/explain``):

1. Is she due? Women screened recently are not.
2. Did she screen before? Lapsed screeners vs. women never screened (in 5 years).
3. Is she connected to preventive care? Gynaecologist prevention, other screening
   programmes, any healthcare contact.

Rules are applied in the order of ``SEGMENTS``; the first that matches wins.

Usage (prints the size and response of each segment):
    uv run python segments.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "data" / "Rakhaton2026_prepared.csv"

DUE_MONTHS = 24  # programme interval: screened less than 24 months ago = not due yet

# name -> short description of the rule (in order of priority)
SEGMENTS = {
    "A not due yet": f"screened < {DUE_MONTHS} months before the invitation",
    "B first invitation": "first invitation of the woman",
    "C lapsed screener, gynaecologist": (
        "screened before (>= 24 months ago), gynaecologist prevention visit in 2 years"
    ),
    "D lapsed screener, no gynaecologist": (
        "screened before (>= 24 months ago), no gynaecologist prevention visit"
    ),
    "E never screened, gynaecologist": (
        "repeat invitation, not screened in 5 years, gynaecologist prevention visit"
    ),
    "F never screened, other screening": (
        "repeat, not screened, no gynaecologist visit, but other screening programme "
        "in 3 years"
    ),
    "G never screened, healthcare user": (
        "repeat, not screened, no prevention, but some healthcare contact in 12 months"
    ),
    "H disconnected": "repeat, not screened, no prevention and no healthcare contact",
}

# segment -> suggested intervention, short (a hypothesis to test, see report.ipynb)
INTERVENTIONS = {
    "A not due yet": "don't send yet; invite ~24 months after the last screen",
    "B first invitation": "standard letter + easy online booking",
    "C lapsed screener, gynaecologist": "reminder at the due date + gynaecologist",
    "D lapsed screener, no gynaecologist": "reminder at the due date + GP",
    "E never screened, gynaecologist": "recommendation by her gynaecologist",
    "F never screened, other screening": "bundle with the screening she already does",
    "G never screened, healthcare user": "personal nudge by GP / pharmacist",
    "H disconnected": "stop letters; phone, navigator or mobile unit",
}

COLUMNS = [
    "FIRST_INVITATION",
    "MONTHS_SINCE_LAST_SCREEN",
    "GYN_PREVENTION_24M_CNT",
    "OTHER_SCREENING_36M_CNT",
    "HEALTHCARE_CONTACT_12M_CNT",
]


def assign(df: pd.DataFrame) -> pd.Series:
    """Segment of every invitation (prepared data: flags are 1/0)."""
    months = df["MONTHS_SINCE_LAST_SCREEN"]
    screened_before = months.notna()
    gyn = df["GYN_PREVENTION_24M_CNT"] > 0
    other = df["OTHER_SCREENING_36M_CNT"] > 0
    contact = df["HEALTHCARE_CONTACT_12M_CNT"] > 0
    conditions = [
        screened_before & (months < DUE_MONTHS),
        df["FIRST_INVITATION"] == 1,
        screened_before & gyn,
        screened_before,
        gyn,
        other,
        contact,
    ]
    names = list(SEGMENTS)
    segment = np.select(conditions, names[:-1], default=names[-1])
    return pd.Series(
        pd.Categorical(segment, categories=names), index=df.index, name="SEGMENT"
    )


def main() -> None:
    labels = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
    df = pd.read_csv(PREPARED, usecols=COLUMNS + labels)
    segment = assign(df)
    summary = df[labels].groupby(segment, observed=False).mean()
    summary.insert(0, "share", segment.value_counts(normalize=True, sort=False))
    summary.insert(0, "invitations", segment.value_counts(sort=False))
    with pd.option_context("display.width", 200):
        print(summary.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()

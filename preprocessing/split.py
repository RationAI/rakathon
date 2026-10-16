"""Out-of-time train / validation / test split of the screening invitations.

Each row is one invitation sent at ``INVITATION_DATE`` (T0) and its labels look up to
365 days past T0. The split is by T0:

    train  T0 <  VAL_START - EMBARGO
    val    VAL_START <= T0 < TEST_START
    test   T0 >= TEST_START

The embargo drops training invitations whose label window would reach into the
validation period, so no training label depends on screenings that happened after
validation starts. The same woman can appear in several splits; that mirrors
deployment, where a model trained on past invitations scores her next one.

The split is deterministic: it uses no randomness and rows are sorted by
(T0, INVITATION_ID), so the same input always gives the same files.

Usage:
    uv run python split.py
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "Rakhaton2026.csv"
OUTPUT = ROOT / "data" / "splits"

VAL_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2024-01-01")
EMBARGO = pd.Timedelta(days=365)  # longest label window (SCREENING_12M_FLAG)


def invitation_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["INVITATION_DATE"], format="%d.%m.%Y")


def load(path: Path) -> pd.DataFrame:
    # Read every value as text so the split files carry the source values unchanged
    # (no "12" -> "12.0", PERSON_ID keeps its leading zeros).
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    order = df.assign(t0=invitation_date(df)).sort_values(["t0", "INVITATION_ID"]).index
    return df.loc[order].reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t0 = invitation_date(df)
    return {
        "train": df[t0 < VAL_START - EMBARGO],
        "val": df[(t0 >= VAL_START) & (t0 < TEST_START)],
        "test": df[t0 >= TEST_START],
    }


def main() -> None:
    df = load(INPUT)
    splits = split(df)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(OUTPUT / f"{name}.csv", index=False)
        print(f"{name:>5}: {len(part):>9,} rows")
    print(f"Dropped by embargo: {len(df) - sum(map(len, splits.values())):,} rows")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()

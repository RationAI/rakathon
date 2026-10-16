"""Patient-level, stratified train / validation / test split.

Women (``PERSON_ID``) are split 70 / 15 / 15 with all their invitations, stratified by
whether the woman was ever screened within 12 months of an invitation. No woman
appears in two splits. Reproducible via a fixed seed.

Usage:
    uv run python split.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "Rakhaton2026.csv"
OUTPUT = ROOT / "data" / "splits"

SEED = 0
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load(path: Path) -> pd.DataFrame:
    # Read every value as text so the split files carry the source values unchanged
    # (no "12" -> "12.0", PERSON_ID keeps its leading zeros).
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ever_screened = df["SCREENING_12M_FLAG"].eq("A").groupby(df["PERSON_ID"]).any()

    train, rest = train_test_split(
        ever_screened.index,
        test_size=FRACTIONS["val"] + FRACTIONS["test"],
        stratify=ever_screened,
        random_state=SEED,
    )
    val, test = train_test_split(
        rest,
        test_size=FRACTIONS["test"] / (FRACTIONS["val"] + FRACTIONS["test"]),
        stratify=ever_screened[rest],
        random_state=SEED,
    )
    women = {"train": train, "val": val, "test": test}
    return {name: df[df["PERSON_ID"].isin(ids)] for name, ids in women.items()}


def main() -> None:
    df = load(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, part in split(df).items():
        part.to_csv(OUTPUT / f"{name}.csv", index=False)
        print(
            f"{name:>5}: {len(part):>9,} rows, {part['PERSON_ID'].nunique():>7,} women"
        )


if __name__ == "__main__":
    main()

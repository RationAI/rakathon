"""Validate the train / validation / test files written by ``split.py``.

Checks that the files exist, carry the source values unchanged, keep every woman in
exactly one split, cover every source row, and match a fresh split of the current
source CSV row for row. Exits with status 1 if any check fails.

Usage:
    uv run python validate_split.py
"""

import sys

import pandas as pd

from split import FRACTIONS, INPUT, OUTPUT, load, split


NAMES = list(FRACTIONS)
LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
TOLERANCE = 0.01  # allowed gap between actual and target share of women

failures: list[str] = []


def invitation_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["INVITATION_DATE"], format="%Y-%m-%d")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    missing = [n for n in NAMES if not (OUTPUT / f"{n}.csv").exists()]
    if missing:
        print(f"[FAIL] missing split files: {missing}; run `uv run python split.py`")
        sys.exit(1)

    source = load(INPUT)
    expected = split(source)
    splits = {
        n: pd.read_csv(OUTPUT / f"{n}.csv", dtype=str, keep_default_na=False)
        for n in NAMES
    }
    n_women = source["PERSON_ID"].nunique()

    for n, part in splits.items():
        print(f"\n{n}: {len(part):,} rows")
        check(f"{n}: columns match source", list(part.columns) == list(source.columns))
        check(f"{n}: INVITATION_ID unique", part["INVITATION_ID"].is_unique)
        check(
            f"{n}: PERSON_ID keeps 15 digits",
            part["PERSON_ID"].str.fullmatch(r"\d{15}").all(),
        )

        share = part["PERSON_ID"].nunique() / n_women
        check(
            f"{n}: share of women close to target",
            abs(share - FRACTIONS[n]) <= TOLERANCE,
            f"{share:.1%} vs {FRACTIONS[n]:.0%}",
        )

        years = invitation_date(part).dt.year.value_counts().sort_index()
        check(
            f"{n}: covers all years",
            years.index.tolist() == sorted(invitation_date(source).dt.year.unique()),
            ", ".join(f"{y}: {c:,}" for y, c in years.items()),
        )

        flags = part[LABELS].eq("1")
        check(
            f"{n}: labels are 1/0 and nested (3M => 6M => 12M)",
            part[LABELS].isin(["1", "0"]).all().all()
            and (~flags.iloc[:, 0] | flags.iloc[:, 1]).all()
            and (~flags.iloc[:, 1] | flags.iloc[:, 2]).all(),
            ", ".join(
                f"{label.split('_')[1]} {flags[label].mean():.1%}" for label in LABELS
            ),
        )

        check(
            f"{n}: identical to a fresh split of the source",
            part.equals(expected[n].reset_index(drop=True)),
        )

    print("\nacross splits")
    women = {n: set(part["PERSON_ID"]) for n, part in splits.items()}
    shared = sum(
        len(women[a] & women[b]) for i, a in enumerate(NAMES) for b in NAMES[i + 1 :]
    )
    check("no woman in more than one split", shared == 0, f"{shared:,} shared")

    ids = pd.concat([splits[n]["INVITATION_ID"] for n in NAMES])
    check("no invitation in more than one split", ids.is_unique)
    check(
        "every source invitation is in a split",
        len(ids) == len(source) and source["INVITATION_ID"].isin(ids).all(),
        f"{len(ids):,} of {len(source):,}",
    )

    print(f"\n{len(failures)} check(s) failed" if failures else "\nAll checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

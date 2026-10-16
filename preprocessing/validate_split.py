"""Validate the train / validation / test files written by ``split.py``.

Checks that the files exist, carry the source values unchanged, respect the date
boundaries and the embargo, do not share invitations, and match a fresh split of the
current source CSV row for row. Exits with status 1 if any check fails.

Usage:
    uv run python validate_split.py
"""

import sys

import pandas as pd

from split import (
    EMBARGO,
    INPUT,
    OUTPUT,
    TEST_START,
    VAL_START,
    invitation_date,
    load,
    split,
)


NAMES = ["train", "val", "test"]
LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
BOUNDS = {
    "train": (None, VAL_START - EMBARGO),
    "val": (VAL_START, TEST_START),
    "test": (TEST_START, None),
}

failures: list[str] = []


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

    for n, part in splits.items():
        print(f"\n{n}: {len(part):,} rows")
        check(f"{n}: columns match source", list(part.columns) == list(source.columns))

        t0 = invitation_date(part)
        lo, hi = BOUNDS[n]
        in_bounds = (lo is None or (t0 >= lo).all()) and (hi is None or (t0 < hi).all())
        check(
            f"{n}: dates within bounds",
            in_bounds,
            f"{t0.min().date()} - {t0.max().date()}",
        )

        check(f"{n}: INVITATION_ID unique", part["INVITATION_ID"].is_unique)
        check(
            f"{n}: PERSON_ID keeps 15 digits",
            part["PERSON_ID"].str.fullmatch(r"\d{15}").all(),
        )

        flags = part[LABELS].eq("A")
        check(
            f"{n}: labels are A/N and nested (3M => 6M => 12M)",
            part[LABELS].isin(["A", "N"]).all().all()
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
    ids = pd.concat([splits[n]["INVITATION_ID"] for n in NAMES])
    check("no invitation in more than one split", ids.is_unique)

    gap = (VAL_START - invitation_date(splits["train"]).max()).days
    check("embargo between train and val", gap > EMBARGO.days, f"{gap} days")

    covered = source["INVITATION_ID"].isin(ids)
    dropped_t0 = invitation_date(source[~covered])
    check(
        "only embargoed invitations dropped",
        ((dropped_t0 >= VAL_START - EMBARGO) & (dropped_t0 < VAL_START)).all(),
        f"{(~covered).sum():,} dropped",
    )

    print(f"\n{len(failures)} check(s) failed" if failures else "\nAll checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""Rebuild all derived data from the raw CSV, in order.

1. spatial 01-03  screening centres, area coordinates, area spatial features
                  (web scraping / geocoding: run only if their output is missing,
                  so the first snapshot is kept; delete the file to refresh it)
   external_data  ČSÚ population / foreigner statistics -> data/population/,
                  data/foreigners/ (download; also skipped once present)
2. prepare_data   fixes + engineered features -> data/Rakhaton2026_prepared.csv
3. split          patient-level stratified split -> data/splits/{train,val,test}.csv
4. validate_split checks the split files

Each step is an existing script run from its own folder (they use folder-relative
paths). Stops at the first failing step.

Usage:
    uv run python pipeline.py
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPATIAL = ROOT / "features" / "spatial"

# (folder, script, output that makes the step skippable or None to always run)
STEPS = [
    (SPATIAL, "01_scrape_centers.py", DATA / "mamo_centers.csv"),
    (SPATIAL, "02_geocode_areas.py", DATA / "area_coordinates.csv"),
    (SPATIAL, "03_add_spatial_features.py", DATA / "area_spatial_features.csv"),
    (
        ROOT / "features",
        "external_data.py",
        DATA / "foreigners" / "foreign_women_by_citizenship_2024.csv",
    ),
    (ROOT / "features", "prepare_data.py", None),
    (ROOT / "preprocessing", "split.py", None),
    (ROOT / "preprocessing", "validate_split.py", None),
]


def main() -> None:
    if not (DATA / "Rakhaton2026.csv").exists():
        sys.exit(f"Missing raw data: {DATA / 'Rakhaton2026.csv'}")

    for folder, script, output in STEPS:
        name = f"{folder.relative_to(ROOT).as_posix()}/{script}"
        if output is not None and output.exists():
            print(f"=== skip {name} ({output.name} exists)")
            continue
        print(f"=== run {name}", flush=True)
        subprocess.run([sys.executable, script], cwd=folder, check=True)

    print("=== done")


if __name__ == "__main__":
    main()

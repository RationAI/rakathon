"""Download external reference data into data/ (gitignored) and tidy it.

1. ČSÚ population by sex and 5-year age group (open data 130142-25) -> data/population/women40plus.csv
   (women aged 40+ per okres/kraj per year, 2000-2024)
2. ČSÚ foreigners by sex, age and citizenship (open data 290038) -> data/foreigners/foreign_women40plus.csv
   (foreign women aged 40+ per okres per year, 2020-2024; Praha is the okres "Praha"),
   plus foreign_women_by_citizenship_2024.csv (all ages) and population/women_all_ages.csv

Run from this folder: uv run python external_data.py
"""

import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
POP_URL = "https://csu.gov.cz/docs/107508/c64b7c67-6658-4372-ce1f-0023565369a6/130142-25data052025.zip?version=1.0"
FOREIGN_URLS = {  # one ZIP per year, stav k 31.12.
    2020: "https://csu.gov.cz/docs/107508/e111d0ea-e98e-453b-c096-655081aaa950/290038-21data2020.zip?version=1.0",
    2021: "https://csu.gov.cz/docs/107508/db3e2e61-56c2-60b0-f7d7-b6f15d605a9c/290038-22data2021.zip?version=1.0",
    2022: "https://csu.gov.cz/docs/107508/5383fd9a-05df-efbd-a603-21c814d97e54/290038-23data2022.zip?version=1.0",
    2023: "https://csu.gov.cz/docs/107508/c517eb96-cf8b-1046-718d-6f7a73aa4e6e/290038-24data2023.zip?version=1.0",
    2024: "https://csu.gov.cz/docs/107508/5af1c8d3-dd12-132f-e37c-22664e865ef1/290038-25data2024.zip?version=1.0",
}


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # the ČSÚ server drops long connections, so let curl resume until the file is complete
    for _ in range(10):
        if subprocess.run(["curl", "-sL", "-C", "-", "-m", "120", "-o", str(dst), url]).returncode == 0:
            return
    raise RuntimeError(f"download failed: {url}")


def population() -> None:
    out = DATA / "population" / "women40plus.csv"
    zip_path = DATA / "population" / "pop.zip"
    fetch(POP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        d = pd.read_csv(z.open(name), low_memory=False)
    d = d[(d.pohlavi_txt == "žena") & d.vek_txt.notna() & d.uzemi_typ.isin(["okres", "kraj"])].copy()
    d["age_from"] = d.vek_txt.str.extract(r"^(?:Od )?(\d+)")[0].astype(int)
    d = d[d.age_from >= 40]
    d["year"] = d.casref_do.str[:4].astype(int)
    d.groupby(["uzemi_typ", "vuzemi_txt", "year"]).hodnota.sum().reset_index().to_csv(out, index=False)
    print("wrote", out)
    # women of all ages per okres/kraj (denominator for citizenship groups)
    with zipfile.ZipFile(zip_path) as z:
        full = pd.read_csv(z.open(name), low_memory=False)
    full = full[(full.pohlavi_txt == "žena") & full.vek_txt.notna() & full.uzemi_typ.isin(["okres", "kraj"])]
    full = full.assign(year=full.casref_do.str[:4].astype(int))
    full.groupby(["uzemi_typ", "vuzemi_txt", "year"]).hodnota.sum().reset_index().to_csv(
        DATA / "population" / "women_all_ages.csv", index=False)


def foreigners() -> None:
    """Foreign women aged 40+ per okres (ČSÚ open data 290038, stav k 31.12.), 2020-2024."""
    out = DATA / "foreigners" / "foreign_women40plus.csv"
    rows = []
    for year, url in FOREIGN_URLS.items():
        zip_path = DATA / "foreigners" / f"f{year}.zip"
        fetch(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            d = pd.read_csv(z.open(next(n for n in z.namelist() if n.upper().endswith(".CSV"))), low_memory=False)
        d = d[(d.pohlavi_txt == "žena") & d.stobcan_txt.isna() & d.vek_txt.notna() & (d.vuzemi_cis == 101)].copy()
        d["age_from"] = d.vek_txt.str.extract(r"<(\d+);")[0].astype(int)
        g = d[d.age_from >= 40].groupby("vuzemi_txt").hodnota.sum()
        rows += [{"okres": o, "year": year, "foreign_women_40plus": int(v)} for o, v in g.items()]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "rows")
    # 2024: foreign women (all ages) per okres and citizenship
    with zipfile.ZipFile(DATA / "foreigners" / "f2024.zip") as z:
        d = pd.read_csv(z.open(next(n for n in z.namelist() if n.upper().endswith(".CSV"))), low_memory=False)
    d = d[(d.pohlavi_txt == "žena") & d.stobcan_txt.notna() & d.vek_txt.isna() & (d.vuzemi_cis == 101)]
    d[["vuzemi_txt", "stobcan_txt", "hodnota"]].rename(columns={"vuzemi_txt": "okres", "stobcan_txt": "citizenship", "hodnota": "women"}).to_csv(
        DATA / "foreigners" / "foreign_women_by_citizenship_2024.csv", index=False)
    print("wrote citizenship file")


if __name__ == "__main__":
    population()
    foreigners()

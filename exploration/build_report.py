"""Build exploration/response_report.html: regional evaluation of invitation response, in the style of the NSC
'Regionální hodnocení dostupnosti mamografického screeningu' slides (kraj bar chart + okres map per indicator).

Run: uv run python build_report.py
Needs data/okresy.geojson, data/praha.geojson, data/area_spatial_features.csv, data/mamo_centers.csv,
data/Rakhaton2026_prepared.csv. Output is self-contained.
"""

import json
import math
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
HERE = Path(__file__).parent
SCALE = 60
KX = math.cos(math.radians(49.8))


def key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper().replace(" ", "")


def xy(lon: float, lat: float) -> tuple[float, float]:
    return round((lon - 12.0) * KX * SCALE, 1), round((51.1 - lat) * SCALE, 1)


cols = ["PERSON_ID", "VZP_TERRITORIAL_AREA", "KRAJ", "AGE_BAND", "AGE_BAND_MEAN", "INVITATION_SEQUENCE", "INVITATION_YEAR",
        "SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
df = pd.read_csv(DATA / "Rakhaton2026_prepared.csv", usecols=cols, dtype={"PERSON_ID": str})
for m in (3, 6, 12):
    df[f"p{m}"] = df[f"SCREENING_{m}M_FLAG"]
P = ["p3", "p6", "p12"]

spatial = pd.read_csv(DATA / "area_spatial_features.csv")
spatial.index = [key(a) for a in spatial.VZP_TERRITORIAL_AREA]
agg = df.groupby("VZP_TERRITORIAL_AREA").agg(kraj=("KRAJ", "first"), n=("KRAJ", "size"), **{c: (c, "sum") for c in P})
agg.index = [key(i) for i in agg.index]
fix = {"PRAHA-HL.MESTO": "PRAHA", "OSTRAVA": "OSTRAVA-MESTO"}
agg = agg.rename(index=fix)
spatial = spatial.rename(index=fix)

pop = pd.read_csv(DATA / "population" / "women40plus.csv")
pop = pop[pop.year.between(2020, 2024)].groupby(["uzemi_typ", "vuzemi_txt"]).hodnota.mean()  # mean women aged 40+ 2020-2024
pop24 = pd.read_csv(DATA / "population" / "women40plus.csv").query("year == 2024").set_index(["uzemi_typ", "vuzemi_txt"]).hodnota
fgn = pd.read_csv(DATA / "foreigners" / "foreign_women40plus.csv").query("year == 2024").set_index("okres").foreign_women_40plus
cit = pd.read_csv(DATA / "foreigners" / "foreign_women_by_citizenship_2024.csv")
GROUPS = {"Ukrajina": "Ukraine", "Slovensko": "Slovakia", "Vietnam": "Vietnam", "Rusko": "Russia", "Polsko": "Poland",
          "Rumunsko": "Romania", "Mongolsko": "Mongolia", "Bulharsko": "Bulgaria"}
cit["group"] = cit.citizenship.map(GROUPS).fillna("Other")
cit = cit.groupby(["okres", "group"]).women.sum().unstack(fill_value=0)
wall = pd.read_csv(DATA / "population" / "women_all_ages.csv").query("year == 2024").set_index(["uzemi_typ", "vuzemi_txt"]).hodnota
KRAJ_OF_MISSING = {"Bruntál": "Moravskoslezský kraj", "Jeseník": "Olomoucký kraj"}  # not VZP areas in the data; map display only

feats = json.load(open(DATA / "okresy.geojson"))["features"] + json.load(open(DATA / "praha.geojson"))["features"]
areas = []
for f in feats:
    name = f["properties"]["nazev"]
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    d = "".join("M" + "L".join(f"{x},{y}" for x, y in (xy(*pt) for pt in r)) + "Z" for p in polys for r in p)
    k = key(name.replace("Hlavní město ", ""))
    a = {"name": "Praha" if k == "PRAHA" else name, "d": d, "kraj": None, "n": None, "p": None, "dist": None, "wait": None,
         "pop": float(pop["kraj", "Hlavní město Praha"] if k == "PRAHA" else pop["okres", name]), "kraj2": KRAJ_OF_MISSING.get(name),
         "forn": int(fgn["Praha" if k == "PRAHA" else name]),
         "eth": {g: int(cit.loc["Praha" if k == "PRAHA" else name, g]) for g in [*GROUPS.values(), "Other"]},
         "wall": float(wall["kraj", "Hlavní město Praha"] if k == "PRAHA" else wall["okres", name]),
         "pop24": float(pop24["kraj", "Hlavní město Praha"] if k == "PRAHA" else pop24["okres", name])}
    if k in agg.index:
        r, s = agg.loc[k], spatial.loc[k]
        a["kraj2"] = r.kraj
        a.update(kraj=r.kraj, n=int(r.n), p=[int(r[c]) for c in P], dist=round(float(s.SPATIAL_DIST_NEAREST_KM), 1),
                 wait=None if pd.isna(s.SPATIAL_WAIT_MEAN_NEAREST3_DAYS) else round(float(s.SPATIAL_WAIT_MEAN_NEAREST3_DAYS), 1))
    areas.append(a)
assert sum(a["n"] is not None for a in areas) == len(agg)

centres = pd.read_csv(DATA / "mamo_centers.csv")
cen = [{"name": r["name"], "city": r.city, "kraj": r.region, "x": xy(r.lon, r.lat)[0], "y": xy(r.lon, r.lat)[1],
        "wait": None if pd.isna(r.wait_days_primary) else float(r.wait_days_primary)} for _, r in centres.iterrows()]

# who responds: kraj x dimension
df["order"] = df.INVITATION_SEQUENCE.clip(upper=5)
age_order = df.drop_duplicates("AGE_BAND").sort_values("AGE_BAND_MEAN").AGE_BAND.tolist()


def breakdown(col: str, labels: dict | None = None) -> dict:
    out = {}
    for kraj, sub in [("ALL", df)] + list(df.groupby("KRAJ")):
        g = sub.groupby(col).agg(n=(col, "size"), **{c: (c, "sum") for c in P})
        out[kraj] = [{"k": (labels or {}).get(i, str(i)), "n": int(r.n), "p": [int(r[c]) for c in P]} for i, r in g.iterrows()]
    return out


who = {"age": breakdown("AGE_BAND"), "order": breakdown("order", {5: "5th+"}), "year": breakdown("INVITATION_YEAR")}
for k, rows in who["age"].items():
    rows.sort(key=lambda r: age_order.index(r["k"]))
for k, rows in who["order"].items():
    for r in rows:
        if r["k"] in "1234":
            r["k"] = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th"}[r["k"]]

meta = {"women": int(df.PERSON_ID.nunique()), "rows": len(df), "centres": len(centres),
        "kraje": sorted(df.KRAJ.unique()), "scraped": str(centres.scraped.max())}
data = {"areas": areas, "centres": cen, "who": who, "meta": meta, "groups": [*GROUPS.values(), "Other"], "w": round(7.1 * KX * SCALE, 1), "h": round(2.6 * SCALE, 1)}
html = (HERE / "response_report.template.html").read_text(encoding="utf-8")
out = HERE / "response_report.html"
out.write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e6:.2f} MB", meta)

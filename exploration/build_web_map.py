"""Build exploration/response_map.html: interactive map of screening response by district and kraj.

Run: uv run python build_web_map.py   (needs data/okresy.geojson, data/praha.geojson, data/Rakhaton2026_prepared.csv)
The HTML is self-contained (data and geometry are embedded).
"""

import json
import math
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).with_name("response_map.html")
LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
SCALE = 60  # svg units per degree of latitude


def key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper().replace(" ", "")


df = pd.read_csv(DATA / "Rakhaton2026_prepared.csv", usecols=["VZP_TERRITORIAL_AREA", "KRAJ", *LABELS])
agg = df.groupby("VZP_TERRITORIAL_AREA").agg(
    kraj=("KRAJ", "first"), n=("KRAJ", "size"), **{f"p{m}": (f"SCREENING_{m}M_FLAG", "sum") for m in (3, 6, 12)}
)
agg.index = [key(i) for i in agg.index]
agg = agg.rename(index={"PRAHA-HL.MESTO": "PRAHA", "OSTRAVA": "OSTRAVA-MESTO"})

feats = json.load(open(DATA / "okresy.geojson"))["features"] + json.load(open(DATA / "praha.geojson"))["features"]
kx = math.cos(math.radians(49.8))


def ring_path(ring: list) -> str:
    pts = [((lon - 12.0) * kx * SCALE, (51.1 - lat) * SCALE) for lon, lat in ring]
    return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"


areas = []
for f in feats:
    name = f["properties"]["nazev"]
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    d = "".join(ring_path(r) for p in polys for r in p)  # outer ring + holes (evenodd)
    k = key(name.replace("Hlavní město ", ""))
    row = agg.loc[k] if k in agg.index else None
    areas.append({
        "name": "Praha" if k == "PRAHA" else name,
        "d": d,
        "kraj": None if row is None else row.kraj,
        "n": None if row is None else int(row.n),
        "p": None if row is None else [int(row.p3), int(row.p6), int(row.p12)],
    })
assert sum(a["n"] is not None for a in areas) == len(agg)

html = TEMPLATE = Path(__file__).with_name("response_map.template.html").read_text(encoding="utf-8")
html = html.replace("__DATA__", json.dumps({"areas": areas, "w": round(7.1 * kx * SCALE, 1), "h": round(2.6 * SCALE, 1)}, ensure_ascii=False, separators=(",", ":")))
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.2f} MB")

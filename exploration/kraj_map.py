"""Choropleth of the screening response rate per district (okres), like the NSC coverage map.

Run: uv run python kraj_map.py [label]    label: SCREENING_3M_FLAG | SCREENING_6M_FLAG | SCREENING_12M_FLAG
Needs data/okresy.geojson + data/praha.geojson (RUIAN, ČÚZK) and data/Rakhaton2026_prepared.csv.
Writes district_response_map.png.
"""

import json
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon

DATA = Path(__file__).resolve().parents[1] / "data"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "SCREENING_12M_FLAG"


def key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper().replace(" ", "")


def rings(geom: dict) -> list:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [p[0] for p in polys]  # outer ring only


df = pd.read_csv(DATA / "Rakhaton2026_prepared.csv", usecols=["VZP_TERRITORIAL_AREA", LABEL])
rate = (df.groupby("VZP_TERRITORIAL_AREA")[LABEL].mean() * 100).rename(index=key)
rate = rate.rename(index={"PRAHA-HL.MESTO": "PRAHA", "OSTRAVA": "OSTRAVA-MESTO"})

feats = json.load(open(DATA / "okresy.geojson"))["features"] + json.load(open(DATA / "praha.geojson"))["features"]
vals = [rate.get(key(f["properties"]["nazev"].replace("Hlavní město ", ""))) for f in feats]
assert sum(v is not None for v in vals) == len(rate), set(rate.index) - {key(f["properties"]["nazev"]) for f in feats}

lo, hi = np.floor(rate.min()), np.ceil(rate.max())
bounds = np.linspace(lo, hi, 5)
cmap = ListedColormap(["#d6e6f7", "#8fb4dc", "#3f78b8", "#16307f"])
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(13, 7.5))
for f, v in zip(feats, vals):
    for ring in rings(f["geometry"]):
        ax.add_patch(Polygon(ring, closed=True, fc="#e0e0e0" if v is None else cmap(norm(v)), ec="white", lw=0.6))
    if v is not None:
        pts = np.array(max(rings(f["geometry"]), key=len))
        x, y = pts.mean(axis=0)
        ax.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if norm(v) >= 2 else "#10204a")
ax.set_xlim(11.9, 19.0)  # Bruntál/Jeseník area is grey: VZP data has no such areas
ax.set_ylim(48.5, 51.1)
ax.set_aspect(1 / np.cos(np.radians(49.8)))
ax.axis("off")
ax.set_title(f"Podíl pozvaných žen, které absolvovaly screening ({LABEL.split('_')[1]}), dle okresu, %",
             fontsize=14, fontweight="bold", color="#2b2d7e")
handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in range(4)]
labels = [f"{a:.1f}–{b:.1f} %" for a, b in zip(bounds[:-1], bounds[1:])] 
handles.append(plt.Rectangle((0, 0), 1, 1, fc="#e0e0e0"))
labels.append("bez dat (Bruntál, Jeseník)")
ax.legend(handles, labels, title="Úspěšnost", loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False)
fig.tight_layout()
out = Path(__file__).with_name("district_response_map.png")
fig.savefig(out, dpi=150)
print("wrote", out, f"overall={df[LABEL].mean()*100:.1f}%", f"range {rate.min():.1f}-{rate.max():.1f}")

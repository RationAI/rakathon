"""Build report/response_report.html: regional evaluation of invitation response, in the style of the NSC
'Regionální hodnocení dostupnosti mamografického screeningu' slides (kraj bar chart + okres map per indicator).

Run: uv run python build_report.py
Needs data/okresy.geojson, data/praha.geojson, data/area_spatial_features.csv, data/mamo_centers.csv,
data/Rakhaton2026_prepared.csv. Output is self-contained.
"""

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data"
HERE = Path(__file__).parent
SCALE = 60
KX = math.cos(math.radians(49.8))


def key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper().replace(" ", "")


def xy(lon: float, lat: float) -> tuple[float, float]:
    return round((lon - 12.0) * KX * SCALE, 1), round((51.1 - lat) * SCALE, 1)


MODEL_FEATURES = [  # who the woman is and her history; no area or spatial feature, so the area effect stays in the residual
    "AGE_BAND_MEAN", "FIRST_INVITATION", "INVITATION_SEQUENCE", "PRIOR_INVITATION_COUNT", "MONTHS_LAST_INVITATION",
    "HEALTHCARE_CONTACT_12M_CNT", "INSURANCE_DURATION_YEARS", "GP_REGISTERED", "GYN_REGISTERED_FLAG",
    "GP_REG_DURATION_BAND_ORD", "GYN_REG_DURATION_BAND_ORD", "PRIOR_SCREEN_COUNT_5Y", "MONTHS_SINCE_LAST_SCREEN",
    "NEVER_SCREENED_BEFORE_FLAG", "GP_PREVENTION_24M_CNT", "GYN_PREVENTION_24M_CNT", "OTHER_SCREENING_36M_CNT",
    "DENTAL_PREVENTION_24M_CNT", "INVITATION_YEAR", "INVITATION_MONTH"]
cols = ["PERSON_ID", "INVITATION_ID", "INVITATION_DATE", "VZP_TERRITORIAL_AREA", "KRAJ", "AGE_BAND", "INVITATION_SEQUENCE",
        "SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG", *MODEL_FEATURES]
df = pd.read_csv(DATA / "Rakhaton2026_prepared.csv", usecols=cols, dtype={"PERSON_ID": str, "INVITATION_ID": str})
for m in (3, 6, 12):
    df[f"p{m}"] = df[f"SCREENING_{m}M_FLAG"]


def expected_12m(frame: pd.DataFrame) -> pd.Series:
    """Out-of-fold probability of the 12 month response from the woman's own features (3 folds, split by woman)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    ids = frame.PERSON_ID.unique()
    fold = frame.PERSON_ID.map(pd.Series(np.random.default_rng(0).integers(0, 3, len(ids)), index=ids)).to_numpy()
    out = np.zeros(len(frame))
    for k in range(3):
        train = fold != k
        model = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
            frame.loc[train, MODEL_FEATURES], frame.loc[train, "SCREENING_12M_FLAG"])
        out[~train] = model.predict_proba(frame.loc[~train, MODEL_FEATURES])[:, 1]
    return pd.Series(out, index=frame.index)


df["pred"] = expected_12m(df)
# cluster-robust variance of (observed - expected): residuals are summed per woman first, because one woman can have up to 5 rows
df["res"] = df.SCREENING_12M_FLAG - df.pred
_w = df.groupby("PERSON_ID").agg(area=("VZP_TERRITORIAL_AREA", "first"), res=("res", "sum"))
cluster_var = (_w.res ** 2).groupby(_w.area).sum()
P = ["p3", "p6", "p12"]

spatial = pd.read_csv(DATA / "area_spatial_features.csv")
spatial.index = [key(a) for a in spatial.VZP_TERRITORIAL_AREA]
df["order"] = df.INVITATION_SEQUENCE.clip(upper=5)  # 1st .. 4th, 5th and later
by_order = df.groupby(["VZP_TERRITORIAL_AREA", "order"])[P].sum().join(df.groupby(["VZP_TERRITORIAL_AREA", "order"]).size().rename("n"))
# per woman: at least one screening within the window after any of her invitations
per_woman = df.groupby("PERSON_ID").agg(area=("VZP_TERRITORIAL_AREA", "first"), **{c: (c, "max") for c in P})
women = per_woman.groupby("area").agg(wn=("p3", "size"), **{f"w{m}": (f"p{m}", "sum") for m in (3, 6, 12)})
women.index = [fix_key for fix_key in (key(i) for i in women.index)]
agg = df.groupby("VZP_TERRITORIAL_AREA").agg(kraj=("KRAJ", "first"), n=("KRAJ", "size"), e=("pred", "sum"),
                                              **{c: (c, "sum") for c in P}).join(cluster_var.rename("ev"))
agg.index = [key(i) for i in agg.index]
fix = {"PRAHA-HL.MESTO": "PRAHA", "OSTRAVA": "OSTRAVA-MESTO"}
agg = agg.rename(index=fix)
by_order.index = pd.MultiIndex.from_tuples([(fix.get(key(a), key(a)), o) for a, o in by_order.index])
women = women.rename(index=fix)

# --- reminder funnel: n-th invitation of the woman (rows are sorted by woman and date), women not yet screened in 12 months ---
df = df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"]).reset_index(drop=True)
df["round"] = df.groupby("PERSON_ID").cumcount() + 1
df["earlier_ok"] = (df.groupby("PERSON_ID").p12.cumsum() - df.p12) > 0
risk = df[(df["round"] <= 5) & ~df.earlier_ok]
funnel = risk.groupby(["VZP_TERRITORIAL_AREA", "round"])[P].sum().join(risk.groupby(["VZP_TERRITORIAL_AREA", "round"]).size().rename("n"))
funnel.index = pd.MultiIndex.from_tuples([(fix.get(key(a), key(a)), r) for a, r in funnel.index])

# --- timeline: invitation date x area, all invitations and 1st invitations ---
dates = sorted(df.INVITATION_DATE.unique())
df["di"] = df.INVITATION_DATE.map({d: i for i, d in enumerate(dates)})
df["first"] = (df.order == 1)


def timeline(frame: pd.DataFrame) -> dict:
    g = frame.groupby(["VZP_TERRITORIAL_AREA", "di"])[P].sum().join(frame.groupby(["VZP_TERRITORIAL_AREA", "di"]).size().rename("n"))
    out = {}
    for (a, di), r in g.iterrows():
        out.setdefault(fix.get(key(a), key(a)), [0] * (4 * len(dates)))[4 * di:4 * di + 4] = [int(r.n), *[int(r[c]) for c in P]]
    return out


tl_all, tl_first = timeline(df), timeline(df[df["first"]])

# --- heatmap: kraj x age band, all and 1st invitations ---
age_order = df.drop_duplicates("AGE_BAND").sort_values("AGE_BAND_MEAN").AGE_BAND.tolist()
age_bands = age_order
heat = {}
for flag, frame in (("all", df), ("first", df[df["first"]])):
    g = frame.groupby(["KRAJ", "AGE_BAND"])[P].sum().join(frame.groupby(["KRAJ", "AGE_BAND"]).size().rename("n"))
    for (k, a), r in g.iterrows():
        heat.setdefault(k, {}).setdefault(a, {})[flag] = [int(r.n), *[int(r[c]) for c in P]]


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
centres = pd.read_csv(DATA / "mamo_centers.csv")
cw = centres.dropna(subset=["lat", "lon", "wait_days_primary"])


def centroid(ring: list) -> tuple[float, float]:
    """Area centroid (shoelace) of a lon/lat ring."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        c = x0 * y1 - x1 * y0
        a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    return cx / (3 * a), cy / (3 * a)


def centre_access(lon: float, lat: float) -> tuple[float, float]:
    """Straight-line km to the nearest centre and mean booking wait of the 3 nearest (as in spatial/03)."""
    la1, lo1, la2, lo2 = map(np.radians, (lat, lon, cw.lat.to_numpy(), cw.lon.to_numpy()))
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    o = np.argsort(d)
    return float(d[o[0]]), float(cw.wait_days_primary.to_numpy()[o[:3]].mean())


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
        a["e"], a["ev"] = round(float(r.e), 1), round(float(r.ev), 1)
        a["wn"], a["wp"] = int(women.loc[k, "wn"]), [int(women.loc[k, f"w{m}"]) for m in (3, 6, 12)]
        a["fn"] = [[int(funnel.loc[(k, r), "n"]), *[int(funnel.loc[(k, r), c]) for c in P]] for r in range(1, 6)]
        a["tl"] = [tl_all.get(k, [0] * 4 * 60), tl_first.get(k, [0] * 4 * 60)]
        a["ord"] = [{"n": int(by_order.loc[(k, o), "n"]), "p": [int(by_order.loc[(k, o), c]) for c in P]} for o in range(1, 6)]
        a.update(kraj=r.kraj, n=int(r.n), p=[int(r[c]) for c in P], dist=round(float(s.SPATIAL_DIST_NEAREST_KM), 1),
                 wait=None if pd.isna(s.SPATIAL_WAIT_MEAN_NEAREST3_DAYS) else round(float(s.SPATIAL_WAIT_MEAN_NEAREST3_DAYS), 1))
    if a["dist"] is None:  # not a VZP area (Bruntál, Jeseník): estimate from the district centroid
        ring = max((r for p in polys for r in p), key=len)
        a["dist"], a["wait"] = (round(v, 1) for v in centre_access(*centroid(ring)))
    areas.append(a)
assert sum(a["n"] is not None for a in areas) == len(agg)

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
data = {"areas": areas, "centres": cen, "who": who, "meta": meta, "dates": dates, "heat": heat, "ages": age_bands, "groups": [*GROUPS.values(), "Other"], "w": round(7.1 * KX * SCALE, 1), "h": round(2.6 * SCALE, 1)}
html = (HERE / "response_report.template.html").read_text(encoding="utf-8")
out = HERE / "response_report.html"
out.write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e6:.2f} MB", meta)

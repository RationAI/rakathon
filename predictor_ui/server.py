"""Invitation planner: a small local web app over the predictive model and the segments.

A manager selects women (by region, district, age band, segment, predicted chance, or a
pasted list of PERSON_IDs). For each woman the app shows her predicted chance of screening
within 3 and 12 months, her segment (segmentation/segments.py) and the intervention and A/B
tests proposed for it (report/evidence.json), and exports the selection as an action list.

Data source: today our CSV (data/Rakhaton2026_prepared.csv); each woman is represented by her
latest invitation in it. In production, ``load_women`` is the one place to replace with a query
to VZP's data, returning the same columns as of the planned mailing date.

Model: gradient boosting on the 19 raw attributes (training/classifiers), one model per label,
fitted on the train split on first start and cached in app/model.joblib (delete it to refit).
Women from the train split are scored in-sample, so their chances are slightly optimistic.

Run: uv run python server.py   (from app/), then open http://127.0.0.1:8000
"""

import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "training" / "classifiers"))
sys.path.insert(0, str(ROOT / "segmentation"))
from segments import BARRIERS, COLUMNS, DUE_MONTHS, INTERVENTIONS, SEGMENTS, assign

from data import CATEGORICAL, FEATURES, features, load
from gradient_boosting.model import make_model


PORT = 8000
MODEL = HERE / "model.joblib"
LABELS = {"p3": "SCREENING_3M_FLAG", "p12": "SCREENING_12M_FLAG"}
PAGE_SIZE = 100


# ---------------------------------------------------------------- model


def fit_or_load() -> dict:
    if MODEL.exists():
        return joblib.load(MODEL)
    print("Fitting the model on the train split (first start only, a few minutes)...", flush=True)
    train = load("train")
    categories = {col: sorted(train[col].dropna().unique()) for col in CATEGORICAL}
    x = features(train, categories)
    bundle = {"categories": categories,
              "models": {key: make_model(x).fit(x, train[label]) for key, label in LABELS.items()}}
    joblib.dump(bundle, MODEL)
    return bundle


# ---------------------------------------------------------------- data


def load_women() -> pd.DataFrame:
    """One row per woman with the model's inputs. Replace with the VZP query in production."""
    cols = ["PERSON_ID", "INVITATION_ID", "INVITATION_DATE", "KRAJ", *FEATURES, *COLUMNS]
    df = pd.read_csv(ROOT / "data" / "Rakhaton2026_prepared.csv", usecols=list(dict.fromkeys(cols)),
                     dtype={"INVITATION_ID": str, "PERSON_ID": str}, keep_default_na=False, na_values=[""])
    df = df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"])
    return df.drop_duplicates("PERSON_ID", keep="last").reset_index(drop=True)


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    x = features(df, bundle["categories"])
    out = df[["PERSON_ID", "INVITATION_DATE", "KRAJ", "VZP_TERRITORIAL_AREA", "AGE_BAND"]].copy()
    for key, model in bundle["models"].items():
        out[key] = model.predict_proba(x)[:, 1]
    out["segment"] = assign(df).astype(str)
    # not due yet: suggest a date ~24 months after her last screen
    last = pd.to_datetime(df.INVITATION_DATE) - pd.to_timedelta(df.MONTHS_SINCE_LAST_SCREEN * 30.44, unit="D")
    due = (last + pd.to_timedelta(DUE_MONTHS * 30.44, unit="D")).dt.strftime("%Y-%m")
    out["note"] = np.where(out.segment == "A not due yet", "postpone to " + due.fillna(""), "")
    return out


print("Loading data...", flush=True)
BUNDLE = fit_or_load()
WOMEN = score(load_women(), BUNDLE)
EVIDENCE = json.loads((ROOT / "report" / "evidence.json").read_text(encoding="utf-8"))
TEST_NAMES = {m["key"]: m["name"] for m in EVIDENCE["mechanisms"]}
OPTIONS = {
    "kraje": sorted(WOMEN.KRAJ.dropna().unique()),
    "areas": WOMEN.groupby("VZP_TERRITORIAL_AREA").KRAJ.first().sort_index().to_dict(),
    "ages": sorted(WOMEN.AGE_BAND.dropna().unique()),
    "segments": [{"k": s, "letter": s[0], "name": s.split(" ", 1)[1], "rule": rule,
                  "intervention": INTERVENTIONS[s], "barrier": BARRIERS[s][0],
                  "tests": [TEST_NAMES[t] for t in BARRIERS[s][1]]} for s, rule in SEGMENTS.items()],
    "women": len(WOMEN),
}
print(f"Ready: {len(WOMEN):,} women. Open http://127.0.0.1:{PORT}", flush=True)


# ---------------------------------------------------------------- selection


def select(q: dict) -> pd.DataFrame:
    ids = [i.strip() for i in q.get("ids", "").replace(",", "\n").split() if i.strip()]
    lists = ["kraje", "areas", "ages", "segments"]
    narrowed = q.get("min", 0) > 0 or q.get("max", 100) < 100
    if not (q.get("all") or ids or narrowed or any(q.get(k) for k in lists)):
        return WOMEN.iloc[:0]  # nothing chosen yet: select no-one
    keep = pd.Series(True, index=WOMEN.index)
    if ids:
        keep &= WOMEN.PERSON_ID.isin(ids)
    for key, col in [("kraje", "KRAJ"), ("areas", "VZP_TERRITORIAL_AREA"), ("ages", "AGE_BAND"), ("segments", "segment")]:
        if q.get(key):
            keep &= WOMEN[col].isin(q[key])
    keep &= WOMEN.p12.between(q.get("min", 0) / 100, q.get("max", 100) / 100)
    return WOMEN[keep]


def summary(sel: pd.DataFrame) -> dict:
    by = sel.groupby("segment").agg(n=("p12", "size"), p12=("p12", "mean"), p3=("p3", "mean"))
    hist = {k: np.histogram(sel[k], bins=20, range=(0, 1))[0].tolist() for k in ("p3", "p12")}  # 5 pp bins
    return {"n": len(sel), "expected": float(sel.p12.sum()), "p12": float(sel.p12.mean()) if len(sel) else None,
            "p3": float(sel.p3.mean()) if len(sel) else None, "hist": hist,
            "segments": [{"k": k, "n": int(r.n), "p12": float(r.p12), "p3": float(r.p3)} for k, r in by.iterrows()]}


def page(sel: pd.DataFrame, q: dict) -> list[dict]:
    col = {"chance": "p12", "id": "PERSON_ID", "segment": "segment", "area": "VZP_TERRITORIAL_AREA"}.get(q.get("sort"), "p12")
    sel = sel.sort_values([col, "PERSON_ID"], ascending=[not q.get("desc", False), True])
    start = max(0, int(q.get("page", 0))) * PAGE_SIZE
    rows = sel.iloc[start:start + PAGE_SIZE]
    return [{"id": r.PERSON_ID, "date": r.INVITATION_DATE, "kraj": r.KRAJ, "area": r.VZP_TERRITORIAL_AREA,
             "age": r.AGE_BAND, "segment": r.segment, "p3": round(r.p3, 4), "p12": round(r.p12, 4), "note": r.note}
            for r in rows.itertuples()]


def export(sel: pd.DataFrame) -> bytes:
    out = sel.sort_values("p12").assign(
        intervention=lambda d: d.segment.map(INTERVENTIONS),
        tests=lambda d: d.segment.map(lambda s: "; ".join(TEST_NAMES[t] for t in BARRIERS[s][1])),
        p3=lambda d: d.p3.round(3), p12=lambda d: d.p12.round(3))
    out = out.rename(columns={"INVITATION_DATE": "LAST_INVITATION_DATE", "p3": "CHANCE_3M", "p12": "CHANCE_12M",
                              "segment": "SEGMENT", "intervention": "INTERVENTION", "tests": "AB_TESTS", "note": "NOTE"})
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel reads the Czech names


# ---------------------------------------------------------------- server


def html() -> bytes:
    page_ = (HERE / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "report" / "site.css").read_text(encoding="utf-8")
    return page_.replace("/*__SITE_CSS__*/", css).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send(html(), "text/html; charset=utf-8")
        elif self.path == "/api/options":
            self.send(json.dumps(OPTIONS, ensure_ascii=False).encode("utf-8"), "application/json")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        sel = select(q)
        if self.path == "/api/select":
            body = {**summary(sel), "rows": page(sel, q), "page_size": PAGE_SIZE}
            self.send(json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")
        elif self.path == "/api/export":
            self.send(export(sel), "text/csv; charset=utf-8",
                      {"Content-Disposition": 'attachment; filename="action_list.csv"'})
        else:
            self.send_error(404)

    def log_message(self, *args: object) -> None:  # keep the console quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

"""Build report/index.html: interventions and tests, the segments of our data joined with the published evidence.

Answers the brief's 03 WHAT and 04 HOW TO VERIFY in one page. Each of the eight segments
(segmentation/segments.py) gets its likely barrier, the intervention and the matching A/B
tests; the tests, protocol, system recommendations and 31 annotated sources come from the
evidence review (evidence.json). The sample-size planner uses each segment's response and
its 2024 invitation volume from the prepared data.

Run: uv run python build_interventions.py   (needs data/Rakhaton2026_prepared.csv)
"""

import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "segmentation"))
from segments import COLUMNS, INTERVENTIONS, SEGMENTS, assign


LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
# segment -> (the likely barrier, in a few words; tests from evidence.json that address it)
BARRIERS = {
    "A not due yet": ("Screened less than two years ago: the letter comes too early or her record is out of date.",
                      ["records"]),
    "B first invitation": ("First letter: she has to find a centre, book, and face an unfamiliar test.",
                           ["booking", "fear"]),
    "C lapsed screener, gynaecologist": ("A regular screener who slipped past her date; she still sees her gynaecologist.",
                                         ["phone", "reminder", "trust"]),
    "D lapsed screener, no gynaecologist": ("A regular screener who slipped past her date, with no gynaecologist visit.",
                                            ["phone", "reminder", "booking"]),
    "E never screened, gynaecologist": ("Sees a gynaecologist for prevention but has not had a mammogram in 5 years.",
                                       ["phone", "trust", "need"]),
    "F never screened, other screening": ("Takes part in another screening programme but skips mammography.",
                                          ["need", "booking"]),
    "G never screened, healthcare user": ("Uses healthcare but no prevention: screening does not feel relevant.",
                                          ["phone", "need", "trust"]),
    "H disconnected": ("No contact with healthcare in a year: letters alone do not reach her.", ["phone", "access", "records"]),
}

df = pd.read_csv(ROOT / "data" / "Rakhaton2026_prepared.csv", usecols=[*COLUMNS, *LABELS, "INVITATION_DATE",
                                                                       "INVITATION_SEQUENCE"])
seg = assign(df)
year = pd.to_datetime(df.INVITATION_DATE).dt.year
segments = []
for s, g in df.groupby(seg, observed=False):
    barrier, tests = BARRIERS[s]
    segments.append({"letter": s[0], "k": s.split(" ", 1)[1], "rule": SEGMENTS[s], "intervention": INTERVENTIONS[s],
                     "barrier": barrier, "tests": tests, "n": len(g), "share": len(g) / len(df),
                     "rates": [float(g[c].mean()) for c in LABELS],
                     "per_arm_year": int((year[g.index] == 2024).sum() // 2)})

rates = [float(df[c].mean()) for c in LABELS]
e = df[seg == "E never screened, gynaecologist"]
e_decay = e.groupby(e.INVITATION_SEQUENCE.clip(upper=9))["SCREENING_12M_FLAG"].mean()
data = {
    "meta": {"rows": len(df), "rates": rates, "late_share": (rates[2] - rates[1]) / rates[2],
             "e_second": float(e_decay.loc[2]), "e_ninth": float(e_decay.loc[9])},
    "segments": segments,
    **json.loads((HERE / "evidence.json").read_text(encoding="utf-8")),
}
html = (HERE / "interventions.template.html").read_text(encoding="utf-8")
html = html.replace("/*__SITE_CSS__*/", (HERE / "site.css").read_text(encoding="utf-8"))
out = HERE / "index.html"
out.write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB")

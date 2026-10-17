"""Build exploration/results.html: model, segments, interventions and experiment as one page.

Companion to exploration.html (same style and navigation). Collects the saved results of
training/classifiers (explain/, ablation/), training/per_group_experts and the segment
rules (segmentation/), fits the gradient-boosting model once for the risk deciles, and
embeds everything into results.template.html. All model numbers are for the 12-month
label (screened within 12 months of the invitation), as in the presentation.

Run: uv run python build_results.py   (~1 min; needs the prepared data and the splits)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASSIFIERS = ROOT / "training" / "classifiers"
sys.path[:0] = [str(CLASSIFIERS), str(ROOT / "segmentation")]
from segments import COLUMNS, INTERVENTIONS, SEGMENTS, assign
from tree_diagram import EDGES, POSITIONS, QUESTIONS, node_masks

from data import prepare
from gradient_boosting.model import make_model


LABEL = "SCREENING_12M_FLAG"
LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", LABEL]
MODEL_NAMES = {
    "gradient boosting": "Gradient boosting", "random forest": "Random forest", "MLP": "Neural network (MLP)",
    "decision tree": "Decision tree", "spline logistic regression": "Spline logistic regression",
    "logistic regression": "Logistic regression", "kNN": "k-nearest neighbours", "naive Bayes": "Naive Bayes",
}
FEATURE_NAMES = {
    "MONTHS_SINCE_LAST_SCREEN": "Months since last screen", "INVITATION_SEQUENCE": "Invitation number (current run)",
    "PRIOR_INVITATION_COUNT": "Previous invitations", "GYN_PREVENTION_24M_CNT": "Gynaecologist prevention (2 y)",
    "OTHER_SCREENING_36M_CNT": "Other screening (3 y)", "MONTHS_LAST_INVITATION": "Months since last invitation",
    "HEALTHCARE_CONTACT_12M_CNT": "Healthcare contacts (12 m)", "DENTAL_PREVENTION_24M_CNT": "Dental prevention (2 y)",
    "AGE_BAND": "Age band", "GYN_REG_DURATION_BAND": "Gynaecologist registration", "VZP_TERRITORIAL_AREA": "VZP area",
    "PRIOR_SCREEN_COUNT_5Y": "Screens in last 5 years", "NEVER_SCREENED_BEFORE_FLAG": "Never screened (5 y)",
    "FIRST_INVITATION": "First invitation", "INSURANCE_DURATION_YEARS": "Insurance duration",
}

# ---------------------------------------------------------------- model: risk deciles
print("Fitting gradient boosting for the risk deciles...", flush=True)
data, x = prepare()
model = make_model(x["train"]).fit(x["train"], data["train"][LABEL])
p = model.predict_proba(x["test"])[:, 1]
y = data["test"][LABEL].to_numpy()
decile = pd.qcut(p, 10, labels=False) + 1  # 1 = lowest predicted participation = highest risk
deciles = [{"k": int(d), "n": int(g.size), "rate": float(g.mean()), "pred": float(p[decile == d].mean())}
           for d, g in pd.Series(y).groupby(decile)]

seg_train, seg_test = assign(data["train"]), assign(data["test"])
rates = data["train"][LABEL].groupby(seg_train, observed=False).mean()
segment_auc = float(roc_auc_score(y, seg_test.map(rates).astype(float)))

# ---------------------------------------------------------------- model comparison & ceiling
expl = CLASSIFIERS / "explain" / "results"
models = pd.read_csv(expl / "models.csv")
comparison = [{"k": MODEL_NAMES[m], "auc": float(a)} for m, a in zip(models.model, models.test_roc_auc, strict=True)
              if m in MODEL_NAMES]
comparison.append({"k": "8 segment rules", "auc": segment_auc, "rules": True})

abl = pd.read_csv(CLASSIFIERS / "ablation" / "results" / "ablation.csv")
best = pd.read_csv(CLASSIFIERS / "ablation" / "results" / "best_per_family.csv")
lc_fams = ["gradient boosting", "MLP", "decision tree", "logistic regression"]
lc = abl[abl.study == "learning curve"]
lc = pd.concat([lc, abl[abl.study == "capacity"].merge(lc[["family", "setting"]].drop_duplicates())])
learning = [{"k": MODEL_NAMES[f], "points": [[int(r.train_rows), float(r.test_roc_auc), float(r.train_roc_auc)]
                                             for r in lc[lc.family == f].sort_values("train_rows").itertuples()]}
            for f in lc_fams]
overfit_settings = [("decision tree", "unlimited, leaf 1"), ("random forest", "unlimited, leaf 1"),
                    ("gradient boosting", "255 leaves, 500 iter, no early stop"),
                    ("MLP", "3 layers, 25 epochs, no early stop")]
cap = abl[abl.study == "capacity"].set_index(["family", "setting"])
overfit = [{"k": f"{MODEL_NAMES[f]}: {s}", "train": float(cap.loc[(f, s), "train_roc_auc"]),
            "test": float(cap.loc[(f, s), "test_roc_auc"])} for f, s in overfit_settings]
feats = abl[abl.study == "features"].merge(cap.reset_index(), on=["family", "setting"], suffixes=("_eng", "_raw"))
features_gain = [{"k": MODEL_NAMES[r.family], "raw": float(r.test_roc_auc_raw), "eng": float(r.test_roc_auc_eng)}
                 for r in feats.itertuples()]
ablation = {
    "fits": len(abl), "families": int(abl.family.nunique()),
    "best": [{"k": MODEL_NAMES.get(r.family, r.family), "setting": r.setting, "auc": float(r.test_roc_auc),
              "gap": float(r.gap_to_best), "lo": float(r.gap_low), "hi": float(r.gap_high)} for r in best.itertuples()],
    "learning": learning, "overfit": overfit, "features": features_gain,
}

# ---------------------------------------------------------------- drivers & effects
perm = pd.read_csv(expl / "permutation_importance.csv")
gb = perm[perm.model == "gradient boosting"].set_index("feature").importance.clip(lower=0)
share = (gb / gb.sum()).sort_values(ascending=False).head(10)
drivers = [{"k": FEATURE_NAMES.get(f, f), "share": float(v)} for f, v in share.items()]
pdp = pd.read_csv(expl / "partial_dependence.csv")
pdp = pdp[pdp.model == "gradient boosting"]
effects = [{"k": FEATURE_NAMES[f], "points": [[float(v), float(m)] for v, m in zip(g.value.astype(float), g.mean_prediction, strict=True)]}
           for f, g in pdp.groupby("feature", sort=False)
           if f in ("PRIOR_INVITATION_COUNT", "MONTHS_SINCE_LAST_SCREEN", "GYN_PREVENTION_24M_CNT")]

# ---------------------------------------------------------------- segments
print("Segments...", flush=True)
cols = [*COLUMNS, *LABELS, "INVITATION_SEQUENCE", "INVITATION_DATE", "PERSON_ID"]
df = pd.read_csv(ROOT / "data" / "Rakhaton2026_prepared.csv", usecols=cols, dtype={"PERSON_ID": str})
seg = assign(df)
year = pd.to_datetime(df.INVITATION_DATE).dt.year
segments = []
for s, g in df.groupby(seg, observed=False):
    segments.append({"letter": s[0], "k": s.split(" ", 1)[1], "rule": SEGMENTS[s], "intervention": INTERVENTIONS[s],
                     "n": len(g), "share": float(len(g) / len(df)), "women": int(g.PERSON_ID.nunique()),
                     "rates": [float(g[c].mean()) for c in LABELS]})
masks = node_masks(df)
tree = {"nodes": {node: {"n": int(m.sum()), "rate": float(df.loc[m, LABEL].mean())} for node, m in masks.items()},
        "questions": QUESTIONS, "edges": EDGES, "positions": POSITIONS}
bins = pd.cut(df.INVITATION_SEQUENCE, [1, 2, 3, 4, 6, 8, 11], labels=["2", "3", "4", "5-6", "7-8", "9-11"])
decay = []
for s in list(SEGMENTS)[4:]:
    m = seg == s
    g = df[m].groupby(bins[m], observed=False)[LABEL].agg(["mean", "size"])
    decay.append({"letter": s[0], "k": s.split(" ", 1)[1],
                  "points": [[str(b), float(r["mean"]), int(r["size"])] for b, r in g.iterrows() if r["size"] >= 500]})

# ---------------------------------------------------------------- experiment sizes (+2 p.p., alpha 5 %, power 80 %)
z, delta = 1.959964 + 0.841621, 0.02
experiment = []
for s in list(SEGMENTS)[2:]:
    p0 = float(df.loc[seg == s, LABEL].mean())
    need = z**2 * (p0 * (1 - p0) + (p0 + delta) * (1 - p0 - delta)) / delta**2
    experiment.append({"letter": s[0], "k": s.split(" ", 1)[1], "baseline": p0, "need": int(np.ceil(need)),
                       "available": int(((seg == s) & (year == 2024)).sum() // 2)})

# ---------------------------------------------------------------- per-group experts
groups = pd.read_csv(ROOT / "training" / "per_group_experts" / "results" / "group_scores.csv")
experts = [{"criterion": r.criterion, "k": r.group, "n": int(r.n_test), "rate": float(r.base_rate),
            "global": float(r.auc_global), "expert": float(r.auc_expert), "gain": float(r.gain),
            "lo": float(r.gain_low), "hi": float(r.gain_high)} for r in groups.itertuples() if r.criterion != "all"]

out_data = {
    "meta": {"test_rows": len(y), "auc": float(roc_auc_score(y, p)), "segment_auc": segment_auc,
             "rate": float(y.mean())},
    "deciles": deciles, "comparison": comparison, "ablation": ablation, "drivers": drivers, "effects": effects,
    "segments": segments, "tree": tree, "decay": decay, "experiment": experiment, "experts": experts,
}
html = (HERE / "results.template.html").read_text(encoding="utf-8")
out = HERE / "results.html"
out.write_text(html.replace("__DATA__", json.dumps(out_data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")

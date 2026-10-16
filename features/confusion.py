"""Confusion matrices: train 2020-21, pick F1-optimal threshold on 2022, report on 2024. Also threshold 0.5."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
import history_features as h

df = h.load(); hist = h.add_history(df)
year = df.INVITATION_DATE.dt.year
tr, va, te = year <= 2021, year == 2022, year == 2024
base = [*h.NUM, *h.CAT, "GP_REGISTERED", "GYN_REGISTERED_FLAG", "FIRST_INVITATION", "NEVER_SCREENED_BEFORE_FLAG"]
rows = []
for name, fs in [("baseline", base), ("baseline+history", base + hist)]:
    fs = [f for f in fs if df.loc[tr, f].nunique() > 1]
    for lab in h.LABELS:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, categorical_features="from_dtype",
                                             early_stopping=True, random_state=0).fit(df.loc[tr, fs], df.loc[tr, lab])
        pv = clf.predict_proba(df.loc[va, fs])[:, 1]
        p, r, t = precision_recall_curve(df.loc[va, lab], pv)
        f1 = 2 * p * r / np.maximum(p + r, 1e-9)
        t_opt = t[np.argmax(f1[:-1])]
        pt = clf.predict_proba(df.loc[te, fs])[:, 1]; y = df.loc[te, lab].values
        for tname, thr in [("0.5", 0.5), ("F1-opt", t_opt)]:
            tn, fp, fn, tp = confusion_matrix(y, pt >= thr).ravel()
            rows.append(dict(model=name, label=lab[10:13], thr=tname, t=round(float(thr), 3), TN=tn, FP=fp, FN=fn, TP=tp,
                             precision=tp / max(tp + fp, 1), recall=tp / (tp + fn), flagged=(tp + fp) / len(y)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
out.to_csv("confusion_matrices.csv", index=False)

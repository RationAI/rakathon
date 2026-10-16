"""Permutation importance (drop in ROC AUC on held-out years) for baseline+history, 12M label, split B."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import history_features as h

df = h.load(); hist = h.add_history(df)
year = df.INVITATION_DATE.dt.year
train, test = year <= 2022, year == 2024
base = [*h.NUM, *h.CAT, "GP_REGISTERED", "GYN_REGISTERED_FLAG", "FIRST_INVITATION", "NEVER_SCREENED_BEFORE_FLAG"]
feats = [f for f in base + hist if df.loc[train, f].nunique() > 1]
print("constant dropped:", [f for f in base + hist if f not in feats])
lab = "SCREENING_12M_FLAG"
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, categorical_features="from_dtype",
                                     early_stopping=True, random_state=0).fit(df.loc[train, feats], df.loc[train, lab])
Xt = df.loc[test, feats].sample(80000, random_state=0); yt = df.loc[Xt.index, lab]
r = permutation_importance(clf, Xt, yt, scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1)
out = pd.DataFrame({"feature": feats, "auc_drop": r.importances_mean, "std": r.importances_std}).sort_values("auc_drop", ascending=False)
print(out.round(4).to_string(index=False))
out.to_csv("feature_importance.csv", index=False)

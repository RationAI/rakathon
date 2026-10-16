"""Global decision tree: one tree per label on the prepared splits.

Uses the numeric prepared features (the string categoricals have ordinal / numeric
counterparts; ``LIN_*`` copies are skipped). sklearn trees handle NaN natively.
The depth / leaf size are picked on validation ROC AUC, then the chosen tree is scored
on test and its top levels are printed.

Usage:
    uv run python decision_tree.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
SEED = 0

LABELS = ["SCREENING_3M_FLAG", "SCREENING_6M_FLAG", "SCREENING_12M_FLAG"]
NOT_FEATURES = ["INVITATION_ID", "PERSON_ID", "INVITATION_DATE", *LABELS]
STRING_COLUMNS = ["AGE_BAND", "VZP_TERRITORIAL_AREA", "GP_REG_DURATION_BAND", "GYN_REG_DURATION_BAND"]
DEPTHS = [3, 4, 5, 6, 8, 10, 12]
MIN_LEAVES = [50, 200, 1000]
PRINT_DEPTH = 4


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / f"{name}.csv", dtype={"INVITATION_ID": str, "PERSON_ID": str})


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=NOT_FEATURES + STRING_COLUMNS)
    return x[[c for c in x.columns if not c.startswith("LIN_")]]


def main() -> None:
    data = {name: load(name) for name in ["train", "val", "test"]}
    x = {name: features(df) for name, df in data.items()}
    print(f"features: {x['train'].shape[1]}")

    for label in LABELS:
        y = {name: df[label] for name, df in data.items()}
        best = None
        print(f"\n{label}")
        for depth in DEPTHS:
            for leaf in MIN_LEAVES:
                tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
                tree.fit(x["train"], y["train"])
                auc = roc_auc_score(y["val"], tree.predict_proba(x["val"])[:, 1])
                print(f"  depth {depth:>2}, min leaf {leaf:>4}: val ROC AUC {auc:.3f}")
                if best is None or auc > best[0]:
                    best = (auc, tree, depth, leaf)

        _, tree, depth, leaf = best
        print(f"chosen: depth {depth}, min leaf {leaf}")
        for name in ["val", "test"]:
            p = tree.predict_proba(x[name])[:, 1]
            print(
                f"  {name}: ROC AUC {roc_auc_score(y[name], p):.3f} | "
                f"PR AUC {average_precision_score(y[name], p):.3f} | "
                f"Brier {brier_score_loss(y[name], p):.3f} | base rate {y[name].mean():.3f}"
            )
        importance = pd.Series(tree.feature_importances_, index=x["train"].columns)
        print("  top features:", ", ".join(f"{k} {v:.2f}" for k, v in importance.nlargest(6).items()))

        small = DecisionTreeClassifier(max_depth=PRINT_DEPTH, min_samples_leaf=1000, random_state=SEED)
        small.fit(x["train"], y["train"])
        print(f"  depth-{PRINT_DEPTH} tree, test ROC AUC "
              f"{roc_auc_score(y['test'], small.predict_proba(x['test'])[:, 1]):.3f}")
        if label == "SCREENING_12M_FLAG":
            print(export_text(small, feature_names=list(x["train"].columns), show_weights=True))


if __name__ == "__main__":
    main()

"""Recurrent network (GRU) over each woman's sequence of invitations.

Instead of scoring every invitation on its own, the woman's invitations are ordered by
date and read by a GRU; the prediction for invitation t uses invitations 1..t. Each step
gets the same 19 preprocessed attributes as the MLP, plus:

- the gap in days since her previous invitation in the data,
- the outcomes (3 / 6 / 12 months) of her earlier invitations, but only once their
  window had closed at the date of invitation t (leakage rule of features/history_features.py);
  an outcome that was still open at the previous step is revealed at the first step where
  it is closed.

Two variants are trained: without and with the earlier outcomes. One network predicts all
three labels (three outputs per step); the epoch with the best mean val ROC AUC is kept.
Gradient boosting (gradient_boosting/model.py) is refitted for the same test rows, and
both are also scored by position in the woman's sequence.

The patient-level split keeps every woman's whole sequence in one split.

Usage (from training/classifiers):
    uv run python -m rnn.model      (~10 min on CPU)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from data import LABELS, SEED, prepare, scores
from gradient_boosting.model import make_model as make_gradient_boosting
from mlp.neural_network import make_preprocess


HIDDEN = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024  # women per batch
EPOCHS = 30
PATIENCE = 4
WINDOW_DAYS = [92, 183, 366]  # a 3 / 6 / 12-month outcome is known once this many days have passed
RESULTS = Path(__file__).resolve().parent / "results.csv"


def history_inputs(df: pd.DataFrame, with_outcomes: bool) -> np.ndarray:
    """Extra per-row inputs from the woman's earlier rows (df sorted by woman and date)."""
    woman = df.PERSON_ID.to_numpy()
    day = pd.to_datetime(df.INVITATION_DATE).to_numpy().astype("datetime64[D]").astype(np.int64)
    y = df[LABELS].to_numpy(np.float32)
    n = len(df)
    gap = np.zeros(n, np.float32)
    has_prev = np.zeros(n, np.float32)
    outcome = np.zeros((n, len(LABELS)), np.float32)  # newly known earlier response, per label
    known = np.zeros((n, len(LABELS)), np.float32)  # an earlier outcome became known at this step
    pending: list[tuple[int, int]] = []  # (row, label) outcomes of this woman not yet revealed
    for i in range(n):
        if i == 0 or woman[i] != woman[i - 1]:
            pending = []
        else:
            has_prev[i] = 1
            gap[i] = (day[i] - day[i - 1]) / 365
            pending += [(i - 1, k) for k in range(len(LABELS))]
            still = []
            for row, k in pending:
                if day[i] - day[row] > WINDOW_DAYS[k]:
                    outcome[i, k] = max(outcome[i, k], y[row, k])
                    known[i, k] = 1
                else:
                    still.append((row, k))
            pending = still
    parts = [gap[:, None], has_prev[:, None]]
    if with_outcomes:
        parts += [outcome, known]
    return np.hstack(parts)


def sequences(df: pd.DataFrame) -> np.ndarray:
    """Row indices of every woman's invitations, padded with -1 (df sorted by woman and date)."""
    starts = np.flatnonzero(np.r_[True, df.PERSON_ID.to_numpy()[1:] != df.PERSON_ID.to_numpy()[:-1]])
    lengths = np.diff(np.r_[starts, len(df)])
    index = np.full((len(starts), lengths.max()), -1, np.int64)
    for j in range(lengths.max()):
        has = lengths > j
        index[has, j] = starts[has] + j
    return index


class GRUModel(nn.Module):
    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_inputs, HIDDEN), nn.ReLU())
        self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        self.out = nn.Linear(2 * HIDDEN, len(LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        state, _ = self.gru(h)  # causal: step t only sees steps 1..t
        return self.out(torch.cat([state, h], dim=-1))


def predict(model: nn.Module, rows: torch.Tensor, index: np.ndarray) -> np.ndarray:
    """Per-row probabilities (n_rows x 3)."""
    model.eval()
    out = np.zeros((len(rows), len(LABELS)), np.float32)
    with torch.no_grad():
        for start in range(0, len(index), 8192):
            idx = torch.from_numpy(index[start : start + 8192])
            p = torch.sigmoid(model(rows[idx.clamp(min=0)])).numpy()
            valid = idx.numpy() >= 0
            out[idx.numpy()[valid]] = p[valid]
    return out


def fit(inputs: dict, targets: dict, index: dict) -> GRUModel:
    torch.manual_seed(SEED)
    model = GRUModel(inputs["train"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    y_train = torch.from_numpy(targets["train"])
    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(index["train"]))
        total, count = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(index["train"][order[start : start + BATCH_SIZE]])
            mask = (idx >= 0).float().unsqueeze(-1)
            safe = idx.clamp(min=0)
            loss = (loss_fn(model(inputs["train"][safe]), y_train[safe]) * mask).sum() / (mask.sum() * len(LABELS))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, count = total + loss.item() * mask.sum().item(), count + mask.sum().item()
        p = predict(model, inputs["val"], index["val"])
        auc = float(np.mean([roc_auc_score(targets["val"][:, k], p[:, k]) for k in range(len(LABELS))]))
        print(f"  epoch {epoch:>2}: train loss {total / count:.4f} | val mean ROC AUC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, bad, best_state = auc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def main() -> None:
    data, x = prepare()
    order = {n: df.sort_values(["PERSON_ID", "INVITATION_DATE", "INVITATION_ID"]).index for n, df in data.items()}
    data = {n: df.loc[order[n]].reset_index(drop=True) for n, df in data.items()}
    x = {n: f.loc[order[n]].reset_index(drop=True) for n, f in x.items()}
    preprocess = make_preprocess(x["train"])
    base = {n: preprocess.fit_transform(f) if n == "train" else preprocess.transform(f) for n, f in x.items()}
    targets = {n: df[LABELS].to_numpy(np.float32) for n, df in data.items()}
    index = {n: sequences(df) for n, df in data.items()}
    test = data["test"]
    position = test.groupby("PERSON_ID").cumcount().add(1).clip(upper=3).map({1: "1st in data", 2: "2nd", 3: "3rd+"})
    print(f"Train: {len(data['train']):,} invitations of {len(index['train']):,} women", flush=True)

    predictions = {}
    for name, with_outcomes in [("GRU, invitation sequence", False), ("GRU + earlier outcomes", True)]:
        print(f"\nFitting {name}...", flush=True)
        inputs = {n: torch.tensor(np.hstack([base[n], history_inputs(data[n], with_outcomes)]), dtype=torch.float32)
                  for n in data}
        model = fit(inputs, targets, index)
        predictions[name] = predict(model, inputs["test"], index["test"])

    print("\nFitting gradient boosting (reference, one model per label)...", flush=True)
    predictions["gradient boosting"] = np.column_stack(
        [make_gradient_boosting(x["train"]).fit(x["train"], data["train"][label]).predict_proba(x["test"])[:, 1]
         for label in LABELS])

    rows = []
    for name, p in predictions.items():
        for k, label in enumerate(LABELS):
            rows.append({"model": name, "label": label, "subset": "all test", **scores(test[label], pd.Series(p[:, k]))})
            for pos, part in test.groupby(position):
                rows.append({"model": name, "label": label, "subset": pos, "rows": len(part),
                             "ROC AUC": roc_auc_score(part[label], p[part.index, k])})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS, index=False)
    print("\nTest ROC AUC")
    print(table.pivot_table(index=["label", "subset"], columns="model", values="ROC AUC", sort=False)
          .to_string(float_format="{:.4f}".format))
    print("\nwrote", RESULTS)


if __name__ == "__main__":
    main()

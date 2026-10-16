"""Neural-network baseline: one small MLP with three outputs (3M / 6M / 12M).

Same 19 raw features and the same preprocessing as ``logistic_regression.py`` (one-hot
categoricals, log1p + standardised numerics with missing indicators), fitted on train
only. The network is trained with BCE on the three labels at once; the epoch with the
best mean validation ROC AUC is kept, then scored on val and test.

Usage:
    uv run python neural_network.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn

from data import CATEGORICAL, FLAGS, LABELS, SEED, numeric_features, prepare, scores


HIDDEN = [256, 128]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
EPOCHS = 30
PATIENCE = 5


def make_preprocess(x: pd.DataFrame) -> ColumnTransformer:
    numeric = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        FunctionTransformer(np.log1p),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features(x)),
            ("categorical", categorical, CATEGORICAL),
            ("flags", "passthrough", FLAGS),
        ]
    )


def make_network(n_inputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for n_hidden in HIDDEN:
        layers += [nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Dropout(DROPOUT)]
        n_inputs = n_hidden
    return nn.Sequential(*layers, nn.Linear(n_inputs, len(LABELS)))


def predict(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(inputs)).numpy()


def mean_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(len(LABELS))]))


def main() -> None:
    torch.manual_seed(SEED)
    data, x = prepare()
    preprocess = make_preprocess(x["train"])
    inputs = {
        name: torch.tensor(
            preprocess.fit_transform(f) if name == "train" else preprocess.transform(f),
            dtype=torch.float32,
        )
        for name, f in x.items()
    }
    targets = {name: df[LABELS].to_numpy() for name, df in data.items()}
    y_train = torch.tensor(targets["train"], dtype=torch.float32)
    print(f"Train rows: {len(y_train):,} | inputs: {inputs['train'].shape[1]}")

    model = make_network(inputs["train"].shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, bad_epochs = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y_train))
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(inputs["train"][idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        auc = mean_auc(targets["val"], predict(model, inputs["val"]))
        print(f"epoch {epoch:>2}: train loss {total / len(order):.4f} | val mean ROC AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    results = {}
    for name in ["val", "test"]:
        p = predict(model, inputs[name])
        for i, label in enumerate(LABELS):
            results[(label, name)] = scores(
                data[name][label], pd.Series(p[:, i], index=data[name].index)
            )
    table = pd.DataFrame(results).T
    table.index.names = ["label", "split"]
    print("\nNeural network (best epoch by val)")
    print(table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()

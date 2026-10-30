from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from my_dataset import Preprocessor, StopsDataset

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "outputs" / "my_splits"
N_FOLDS, EPOCHS = 5, 30        # EPOCHS 는 고정: 평가할 fold 를 보고 멈추는 시점을 고르면 안 된다

# test 는 열지 않는다. train 과 val 을 합쳐서 5-fold 로 쓴다.
data = pd.concat([pd.read_parquet(SPLITS / "stops_train.parquet"),
                  pd.read_parquet(SPLITS / "stops_val.parquet")], ignore_index=True)


def average_precision(y, score):
    order = np.argsort(-score)
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return (precision * y).sum() / y.sum()


def make_logreg(n_in):
    return nn.Linear(n_in, 1)


def make_mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))


def fit_and_score(make_model, train_ds, test_ds, seed):
    torch.manual_seed(seed)
    model = make_model(train_ds.X.shape[1])
    n_pos = train_ds.y.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(train_ds) - n_pos) / n_pos)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(test_ds.X).squeeze(1)).numpy()
    return average_precision(test_ds.y.numpy(), score)


models = {"로지스틱 회귀": make_logreg, "MLP": make_mlp}
results = {name: [] for name in models}
base_rates = []

for k in range(N_FOLDS):
    train_df = data[data["cv_fold"] != k]
    held_df = data[data["cv_fold"] == k]
    prep = Preprocessor().fit(train_df)                       # fold 마다 train 부분으로만 다시 배운다
    train_ds, held_ds = StopsDataset(train_df, prep), StopsDataset(held_df, prep)
    base_rates.append(held_ds.y.mean().item())
    for name, make_model in models.items():
        results[name].append(fit_and_score(make_model, train_ds, held_ds, seed=k))
    print(f"fold {k}: " + "  ".join(f"{n} {results[n][-1]:.3f}" for n in models)
          + f"  (무작위 기준선 {base_rates[-1]:.3f})")

print("\n모델            평균 PR-AUC   표준편차   fold 별 최소~최대")
for name, scores in results.items():
    s = np.array(scores)
    print(f"{name:<12}  {s.mean():10.3f}  {s.std(ddof=1):9.3f}   {s.min():.3f} ~ {s.max():.3f}")
print(f"{'무작위':<12}  {np.mean(base_rates):10.3f}")
diff = np.array(results["MLP"]) - np.array(results["로지스틱 회귀"])
print(f"\nMLP - 로지스틱 (fold 별 차이): {np.round(diff, 3)}  평균 {diff.mean():+.3f}")
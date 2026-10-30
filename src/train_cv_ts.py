from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from my_dataset import Preprocessor, StopsDataset, NUMERIC, BINARY, TS_NUMERIC

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "outputs" / "my_splits"
N_FOLDS, EPOCHS = 5, 30

data = pd.concat([pd.read_parquet(SPLITS / "stops_train.parquet"),
                  pd.read_parquet(SPLITS / "stops_val.parquet")], ignore_index=True)
ts = pd.read_parquet(ROOT / "outputs" / "ts_features.parquet")
data = data.merge(ts, on=["scenario_id", "t_onset"], how="left")

# 시계열 특징을 만들 수 있는 정지만 쓴다. 이력이 없는 정지까지 섞으면
# "이력 없음" 자체가 힌트가 되는데, 이는 시나리오 클립 경계 때문이지 주행 상황이 아니다.
data = data[data["has_hist"] == 1]
print(f"사용: {len(data)}건 (급정지 {int(data['hard'].sum())}건, 비율 {data['hard'].mean():.3f})\n")


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


feature_sets = {"기본": NUMERIC, "기본+시계열": NUMERIC + TS_NUMERIC}
models = {"로지스틱 회귀": make_logreg, "MLP": make_mlp}
results = {(f, m): [] for f in feature_sets for m in models}

for k in range(N_FOLDS):
    train_df = data[data["cv_fold"] != k]
    held_df = data[data["cv_fold"] == k]
    for fname, numeric in feature_sets.items():
        prep = Preprocessor(numeric, BINARY).fit(train_df)
        train_ds, held_ds = StopsDataset(train_df, prep), StopsDataset(held_df, prep)
        for mname, make_model in models.items():
            results[(fname, mname)].append(fit_and_score(make_model, train_ds, held_ds, seed=k))
    print(f"fold {k} 끝")

print(f"\n{'특징':<12}{'모델':<14}{'평균 PR-AUC':>11}{'표준편차':>9}   (무작위 {data['hard'].mean():.3f})")
for (fname, mname), scores in results.items():
    s = np.array(scores)
    print(f"{fname:<12}{mname:<14}{s.mean():11.3f}{s.std(ddof=1):9.3f}")

for mname in models:
    diff = np.array(results[("기본+시계열", mname)]) - np.array(results[("기본", mname)])
    print(f"\n{mname}: 시계열 추가 효과 fold 별 {np.round(diff, 3)}  평균 {diff.mean():+.3f}")
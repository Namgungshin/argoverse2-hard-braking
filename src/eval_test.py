from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from my_dataset import Preprocessor, StopsDataset, NUMERIC, BINARY, TS_NUMERIC

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "outputs" / "my_splits"
EPOCHS, SEEDS = 30, range(10)       # 설정은 교차검증 때와 같다. test 결과를 보고 바꾸지 않는다.
TOP_FRACTION = 0.20                 # 점수 상위 20% 에 경보를 울린다고 가정

ts = pd.read_parquet(ROOT / "outputs" / "ts_features.parquet")


def load(names):
    df = pd.concat([pd.read_parquet(SPLITS / f"stops_{n}.parquet") for n in names], ignore_index=True)
    df = df.merge(ts, on=["scenario_id", "t_onset"], how="left")
    return df[df["has_hist"] == 1]          # 시계열 특징을 만들 수 있는 정지만


train_df, test_df = load(["train", "val"]), load(["test"])
print(f"학습 {len(train_df)}건 (급정지 {int(train_df['hard'].sum())}) | "
      f"test {len(test_df)}건 (급정지 {int(test_df['hard'].sum())}, 무작위 기준선 {test_df['hard'].mean():.3f})\n")


def average_precision(y, score):
    order = np.argsort(-score)
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return (precision * y).sum() / y.sum()


def make_logreg(n_in):
    return nn.Linear(n_in, 1)


def make_mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))


def fit_and_predict(make_model, train_ds, test_ds, seed):
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
        return torch.sigmoid(model(test_ds.X).squeeze(1)).numpy()


feature_sets = {"기본": NUMERIC, "기본+시계열": NUMERIC + TS_NUMERIC}
models = {"로지스틱 회귀": make_logreg, "MLP": make_mlp}
y = test_df["hard"].to_numpy()
print(f"{'특징':<10}{'모델':<12}{'평균 PR-AUC':>11}{'최소~최대 (시드 10개)':>24}")
final_scores = None
for fname, numeric in feature_sets.items():
    prep = Preprocessor(numeric, BINARY).fit(train_df)          # 학습 데이터로만 배운다
    train_ds, test_ds = StopsDataset(train_df, prep), StopsDataset(test_df, prep)
    for mname, make_model in models.items():
        scores = [fit_and_predict(make_model, train_ds, test_ds, s) for s in SEEDS]
        aps = np.array([average_precision(y, sc) for sc in scores])
        print(f"{fname:<10}{mname:<12}{aps.mean():11.3f}{aps.min():12.3f} ~ {aps.max():.3f}")
        if (fname, mname) == ("기본+시계열", "MLP"):
            final_scores = np.mean(scores, axis=0)             # 시드 10개의 평균 점수

# 최종 모델(기본+시계열 MLP)이 상위 20% 에 경보를 울린다면
k = int(len(y) * TOP_FRACTION)
flagged = np.argsort(-final_scores)[:k]
caught = int(y[flagged].sum())
print(f"\n[기본+시계열 MLP, 점수 상위 {TOP_FRACTION:.0%} ({k}건)에 경보]")
print(f"잡은 급정지 {caught}/{int(y.sum())}건 → 재현율 {caught / y.sum():.3f}, "
      f"정밀도 {caught / k:.3f} (무작위 {y.mean():.3f})")
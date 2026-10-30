from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from my_dataset import Preprocessor, StopsDataset   # 같은 src 폴더의 my_dataset.py

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)

train_df = pd.read_parquet(ROOT / "outputs" / "my_splits" / "stops_train.parquet")
val_df = pd.read_parquet(ROOT / "outputs" / "my_splits" / "stops_val.parquet")

prep = Preprocessor().fit(train_df)                  # train 으로만 배운다
train_ds, val_ds = StopsDataset(train_df, prep), StopsDataset(val_df, prep)
loader = DataLoader(train_ds, batch_size=128, shuffle=True)
print("입력 차원:", train_ds.X.shape[1], "| 빈 값:", int(torch.isnan(train_ds.X).sum()))

model = nn.Linear(train_ds.X.shape[1], 1)            # 로지스틱 회귀 = 선형층 1개 + 시그모이드

n_pos = train_ds.y.sum()
loss_fn = nn.BCEWithLogitsLoss()       
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)


def average_precision(y, score):
    """PR-AUC: 점수 높은 순으로 줄 세웠을 때, 급정지를 만날 때마다의 정밀도 평균"""
    order = np.argsort(-score)
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return (precision * y).sum() / y.sum()


for epoch in range(1, 61):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()                              # 기울기 초기화
        loss = loss_fn(model(xb).squeeze(1), yb)     # 순전파
        loss.backward()                              # 역전파
        opt.step()                                   # 업데이트

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_logit = model(val_ds.X).squeeze(1)
            val_loss = loss_fn(val_logit, val_ds.y).item()
            val_ap = average_precision(val_ds.y.numpy(), torch.sigmoid(val_logit).numpy())
        print(f"epoch {epoch}  val loss {val_loss:.4f}  val PR-AUC {val_ap:.3f}"
              f"  (무작위 기준선 {val_ds.y.mean():.3f})")

with torch.no_grad():
    prob = torch.sigmoid(model(val_ds.X).squeeze(1)).numpy()
y = val_ds.y.numpy()
pred = prob >= 0.5

print("평균 예측 확률:", round(prob.mean(), 3))
print("급정지로 부른 개수:", int(pred.sum()), "/ 실제 급정지", int(y.sum()))
print("재현율(실제 급정지 중 잡은 비율):", round(pred[y == 1].mean(), 3))
print("정밀도(급정지라 부른 것 중 맞은 비율):", round(y[pred].mean() if pred.any() else 0, 3))

# ---- 기준값(threshold) 실험 ----
with torch.no_grad():
    prob = torch.sigmoid(model(val_ds.X).squeeze(1)).numpy()
y = val_ds.y.numpy()

print(f"\n실제 급정지 {int(y.sum())}건 / 전체 {len(y)}건 (무작위로 부를 때의 정밀도 {y.mean():.3f})")
print("기준  부른수  잡은수  재현율  정밀도  놓친수  헛경보")
for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred = prob >= thr
    called = int(pred.sum())
    caught = int((pred & (y == 1)).sum())
    recall = caught / y.sum()
    precision = caught / called if called else float("nan")
    print(f"{thr:4.2f}  {called:6d}  {caught:6d}  {recall:6.3f}  {precision:6.3f}"
          f"  {int(y.sum()) - caught:6d}  {called - caught:6d}")
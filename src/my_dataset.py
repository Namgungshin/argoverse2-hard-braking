import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUMERIC = ["v_onset", "n_agents_onset", "lead_gap_onset", "lead_rel_onset", "inv_ttc", "near_dist"]
BINARY = ["has_lead", "bus_30m", "vru_20m", "vru_on_crosswalk", "dash_to_solid", "lane_inter"]
CATEGORICAL = ["near_class", "lane_cat"]
TS_NUMERIC = ["dv_1s", "acc_mean_1s", "acc_min_1s", "speed_std_1s"]   # 감속 시작 직전 1초의 움직임


def add_columns(df):
    """행마다 정해지는 변환 (train 통계가 필요 없음)"""
    df = df.copy()
    df["has_lead"] = df["lead_gap_onset"].notna().astype(float)        # 앞차가 있나?
    ttc = df["lead_ttc_onset"]
    df["inv_ttc"] = (1 / ttc.where(ttc > 0)).clip(upper=2).fillna(0)   # 위험할수록 큰 값
    df[["dash_to_solid", "lane_inter"]] = df[["dash_to_solid", "lane_inter"]].fillna(0)
    return df


class Preprocessor:
    """평균·표준편차·범주 목록을 train 에서만 배운다."""

    def __init__(self, numeric=NUMERIC, binary=BINARY):
        self.numeric, self.binary = numeric, binary

    def fit(self, train_df):
        d = add_columns(train_df)
        self.mean = d[self.numeric].mean()
        self.std = d[self.numeric].std()
        self.cats = {c: sorted(d[c].dropna().unique()) for c in CATEGORICAL}
        return self

    def transform(self, df):
        d = add_columns(df)
        num = ((d[self.numeric] - self.mean) / self.std).fillna(0)   # 표준화 후 빈 값은 평균(0)
        onehot = [(d[c] == v).astype(float).rename(f"{c}={v}")
                  for c in CATEGORICAL for v in self.cats[c]]
        X = pd.concat([num, d[self.binary].astype(float)] + onehot, axis=1)
        return X.to_numpy(dtype=np.float32)


class StopsDataset(Dataset):
    def __init__(self, df, prep):
        self.X = torch.tensor(prep.transform(df))
        self.y = torch.tensor(df["hard"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # 프로젝트 폴더 (src 의 한 단계 위)
SEED = 42
N_FOLDS = 5

df = pd.read_parquet(ROOT / "outputs" / "ego_stops_lane.parquet")

# (1) 시나리오 목록: 시나리오마다 "급정지가 하나라도 있나?"
scen = df.groupby("scenario_id")["hard"].max().reset_index()
scen.columns = ["scenario_id", "has_hard"]

# (2) 급정지 있는 시나리오 / 없는 시나리오를 따로 섞어서 70/15/15로 자르기 (층화)
rng = np.random.default_rng(SEED)
split_of = {}
for _, group in scen.groupby("has_hard"):
    ids = rng.permutation(group["scenario_id"].to_numpy())
    n_train = round(len(ids) * 0.70)
    n_val = round(len(ids) * 0.15)
    for sid in ids[:n_train]:
        split_of[sid] = "train"
    for sid in ids[n_train:n_train + n_val]:
        split_of[sid] = "val"
    for sid in ids[n_train + n_val:]:
        split_of[sid] = "test"
        scen["split"] = scen["scenario_id"].map(split_of)

# (2-2) 교차검증 fold: test 를 뺀 시나리오에 0~4 번호를 돌아가며 붙이기 (급정지 유무별로 따로)
scen["cv_fold"] = -1                                   # test 는 -1 로 남긴다
for _, group in scen[scen["split"] != "test"].groupby("has_hard"):
    shuffled = rng.permutation(group.index.to_numpy())
    scen.loc[shuffled, "cv_fold"] = np.arange(len(shuffled)) % N_FOLDS

# (3) 정지 에피소드에 세트 이름 붙이기 -> 같은 시나리오의 정지는 자동으로 같은 세트
by_scenario = scen.set_index("scenario_id")
df["split"] = df["scenario_id"].map(by_scenario["split"])
df["cv_fold"] = df["scenario_id"].map(by_scenario["cv_fold"])

# (4) 누수 검사: 한 시나리오가 두 세트에 걸치면 안 된다
assert (df.groupby("scenario_id")["split"].nunique() == 1).all()
assert (df.groupby("scenario_id")["cv_fold"].nunique() == 1).all()
# (5) 저장 + 세트별 급정지 비율 확인
out = ROOT / "outputs" / "my_splits"
out.mkdir(parents=True, exist_ok=True)
for name in ["train", "val", "test"]:
    part = df[df["split"] == name]
    part.to_parquet(out / f"stops_{name}.parquet", index=False)
    print(name, len(part), "건, 급정지 비율", round(part["hard"].mean(), 3))
# (6) fold 별 확인 (train+val 만)
cv = df[df["split"] != "test"]
print(cv.groupby("cv_fold")["hard"].agg(건수="size", 급정지="sum", 비율="mean").round(3))
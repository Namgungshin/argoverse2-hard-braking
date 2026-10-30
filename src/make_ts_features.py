from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GAP = 5      # 감속 시작 직전 5스텝은 쓰지 않는다 (속도 평활화가 앞쪽 값을 섞어 쓰므로 감속 시작 정보가 새어 들어옴)
WIN = 10     # 1초(10스텝) 구간
MIN_N = 3    # 구간에 이 개수보다 적게 남으면 "이력 없음"으로 본다

stops = pd.read_parquet(ROOT / "outputs" / "ego_stops_lane.parquet", columns=["scenario_id", "t_onset"])
kin = pd.read_parquet(ROOT / "outputs" / "ego_kinematics.parquet",
                      columns=["scenario_id", "timestep", "speed", "acc"])
by_scenario = {sid: g.sort_values("timestep") for sid, g in kin.groupby("scenario_id")}

rows = []
for sid, t_onset in zip(stops["scenario_id"], stops["t_onset"]):
    g = by_scenario[sid]
    end = t_onset - GAP
    w = g[(g["timestep"] > end - WIN) & (g["timestep"] <= end)]
    row = {"scenario_id": sid, "t_onset": t_onset}
    if len(w) >= MIN_N:
        row.update(dv_1s=w["speed"].iloc[-1] - w["speed"].iloc[0],
                   acc_mean_1s=w["acc"].mean(),
                   acc_min_1s=w["acc"].min(),
                   speed_std_1s=w["speed"].std())
    rows.append(row)

feat = pd.DataFrame(rows)
feat["has_hist"] = feat["dv_1s"].notna().astype(float)
feat.to_parquet(ROOT / "outputs" / "ts_features.parquet", index=False)
print("이력 있는 정지:", int(feat["has_hist"].sum()), "/", len(feat))
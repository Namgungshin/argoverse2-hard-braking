"""가설 검증: 자차 급정지/급감속이 어떤 상황에서 생기는가.

H1 끼어들기: 감속 시작 전 3초 ~ 후 0.5초 사이에 옆 차로(횡 3m 이상)에서 같은 방향(진행각 45도 이내, 2 m/s 이상)으로 달리던 차량이 자차 정면 통로(횡 1.8m 이내, 앞 30m)로 들어왔는가
H2 객체 유형: 감속 시작 때 (a) 30m 내 버스, (b) 횡단보도 3m 이내에 있는 보행자·자전거(=건너거나 건너려는 VRU), (c) 20m 내 보행자·자전거
H3 지도 구조: 정지점이 횡단보도 8m 이내 / 교차로 8m 이내 (13번에서 계산)

비교 두 가지
 (1) 정지 에피소드 안에서 급정지(최소 가속도 <= -3) vs 일반 정지 — 감속 시작 속도까지 넣은 로지스틱 회귀(시나리오 클러스터 SE)
 (2) 자차 급감속 이벤트 vs 자차 대조 시점(10번 ego_samples) — 단순 비율 비교

실행: python src/14_ego_hypotheses.py
출력: outputs/ego_stops_feat.parquet, outputs/tables/ego_hypotheses_logit.csv, ego_hypotheses_events.csv
"""
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c = load("c", "src/06_context_features.py")
L = load("L", "src/07_context_analysis.py")
E = load("E", "src/10_ego_analysis.py")
EGO = "AV"
VEH = {"vehicle", "bus", "motorcyclist"}
VRU = {"pedestrian", "cyclist"}
PRE, POST = 30, 5
COLS = c.COLS


def crosswalk_segs(sid):
    m = json.load(open(c.ROOT / sid / f"log_map_archive_{sid}.json"))
    return c.load_map(sid)[2], m


def dist_to_segs(px, py, segs):
    if not segs:
        return np.full(len(px), np.inf)
    out = np.full(len(px), np.inf)
    P = np.column_stack([px, py])
    for a, b in segs:
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        out = np.minimum(out, np.hypot(*(a + t[:, None] * ab - P).T))
    return out


def feats_at(df, by_t, ego, t0, segs):
    """t0(감속 시작)에서의 H1/H2 변수."""
    e0 = ego.loc[t0]
    x, y = float(e0["position_x"]), float(e0["position_y"])
    g = by_t[t0]
    o = g[g["track_id"] != EGO]
    d = np.hypot(o["position_x"].to_numpy() - x, o["position_y"].to_numpy() - y)
    typ = o["object_type"].to_numpy()
    vru = np.isin(typ, list(VRU))
    cw_d = dist_to_segs(o["position_x"].to_numpy()[vru], o["position_y"].to_numpy()[vru], segs) if vru.any() else np.array([])
    out = dict(bus_30m=int(((typ == "bus") & (d <= 30)).sum() > 0),
               vru_20m=int((vru & (d <= 20)).sum() > 0),
               vru_on_crosswalk=int(((cw_d <= 3) & (d[vru] <= 30)).sum() > 0) if vru.any() else 0)
    # H1 끼어들기
    cut = 0
    status = {}
    for t in range(t0 - PRE, t0 + POST + 1):
        if t not in ego.index or t not in by_t:
            continue
        r = ego.loc[t]
        sp = float(np.hypot(r["velocity_x"], r["velocity_y"]))
        if sp < 1.0:
            continue
        ux, uy = r["velocity_x"] / sp, r["velocity_y"] / sp
        gt = by_t[t]
        gt = gt[gt["object_type"].isin(VEH) & (gt["track_id"] != EGO)]
        # 같은 방향으로 달리는 차량만 (정차 차량이나 교차 진행 차량은 자차가 휘어 돌 때 '들어온 것처럼' 보이므로 제외)
        ov = np.hypot(gt["velocity_x"].to_numpy(), gt["velocity_y"].to_numpy())
        cosang = (gt["velocity_x"].to_numpy() * ux + gt["velocity_y"].to_numpy() * uy) / np.maximum(ov, 1e-6)
        gt = gt[(ov >= 2.0) & (cosang >= 0.7)]
        rx = gt["position_x"].to_numpy() - r["position_x"]
        ry = gt["position_y"].to_numpy() - r["position_y"]
        lon, lat = rx * ux + ry * uy, np.abs(rx * uy - ry * ux)
        for tid, lo, la in zip(gt["track_id"].to_numpy(), lon, lat):
            if 0 < lo <= 40 and la >= 3.0:
                status[tid] = "side"
            elif 0 < lo <= 30 and la <= 1.8 and status.get(tid) == "side":
                cut = 1
                break
        if cut:
            break
    out["cut_in"] = cut
    return out


def work(task):
    sid, t0s = task
    df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=COLS)
    by_t = {t: g for t, g in df.groupby("timestep")}
    ego = df[df["track_id"] == EGO].set_index("timestep").sort_index()
    segs = c.load_map(sid)[2]
    rows = []
    for key, t0 in t0s:
        if t0 not in ego.index:
            continue
        r = feats_at(df, by_t, ego, int(t0), segs)
        r.update(scenario_id=sid, key=key)
        rows.append(r)
    return rows


def run_tasks(pairs):
    tasks = {}
    for sid, key, t in pairs:
        tasks.setdefault(sid, []).append((key, int(t)))
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(work, list(tasks.items()), chunksize=20):
            rows += r
    return pd.DataFrame(rows)


def main():
    tdir = Path("outputs/tables")
    S = pd.read_parquet("outputs/ego_stops.parquet").reset_index(drop=True)
    S["key"] = S.index
    F = run_tasks(zip(S["scenario_id"], S["key"], S["t_onset"]))
    S = S.merge(F.drop(columns="scenario_id"), on="key", how="left")
    S.to_parquet("outputs/ego_stops_feat.parquet", index=False)

    S["is_event"] = (S["min_acc"] <= -3.0).astype(float)
    S["speed5"] = S["v_onset"] / 5
    S["near_cross8"] = (S["stop_dist_crosswalk"] <= 8).astype(float)
    S["near_inter8"] = (S["stop_dist_intersection"] <= 8).astype(float)
    S["lead_behind15"] = (S["lead_gap_stop"] <= 15).astype(float)
    cols = ["speed5", "near_cross8", "near_inter8", "lead_behind15", "cut_in", "bus_30m", "vru_on_crosswalk", "vru_20m"]
    lab = {"speed5": "감속 시작 속도 (+5 m/s)", "near_cross8": "H3 정지점 횡단보도 8m 이내", "near_inter8": "H3 정지점 교차로 8m 이내",
           "lead_behind15": "선행 객체 뒤 정지(15m)", "cut_in": "H1 끼어들기", "bus_30m": "H2a 30m 내 버스",
           "vru_on_crosswalk": "H2b 횡단보도 위/옆 보행자·자전거", "vru_20m": "H2c 20m 내 보행자·자전거"}
    X = np.column_stack([np.ones(len(S))] + [S[c_].to_numpy(float) for c_ in cols])
    b, se = L.logit(X, S["is_event"].to_numpy(float), S["scenario_id"].to_numpy())
    res = pd.DataFrame({"급정지비율": [S.loc[S["is_event"] == 1, c_].mean() for c_ in cols],
                        "일반정지비율": [S.loc[S["is_event"] == 0, c_].mean() for c_ in cols],
                        "OR": np.exp(b[1:]), "CI_low": np.exp(b[1:] - 1.96 * se[1:]), "CI_high": np.exp(b[1:] + 1.96 * se[1:])},
                       index=[lab[c_] for c_ in cols]).round(3)
    res.loc[lab["speed5"], ["급정지비율", "일반정지비율"]] = np.nan
    print(f"[1] 정지 에피소드 {len(S):,}건 (급정지 {int(S['is_event'].sum())}) — 로지스틱 회귀, 시나리오 클러스터 강건 SE")
    print(res.to_string())
    res.to_csv(tdir / "ego_hypotheses_logit.csv")

    # 횡단보도 급정지 세부: VRU가 있었나
    cw = S[S["near_cross8"] == 1]
    tab = cw.groupby("is_event")["vru_on_crosswalk"].agg(["size", "mean"]).round(3)
    tab.index = ["일반 정지", "급정지"]
    print("\n[2] 횡단보도 앞 정지만: 감속 시작 때 횡단보도 위/옆 보행자·자전거가 있던 비율\n", tab.to_string())

    # (2) 급감속 이벤트 vs 대조
    d = pd.read_parquet("outputs/ego_samples.parquet").reset_index(drop=True)
    d["key"] = d.index
    G = run_tasks(zip(d["scenario_id"], d["key"], d["t"]))
    d = d.merge(G.drop(columns="scenario_id"), on="key", how="left")
    rows = []
    for name, col in [("H1 끼어들기", "cut_in"), ("H2a 30m 내 버스", "bus_30m"), ("H2b 횡단보도 위/옆 보행자·자전거", "vru_on_crosswalk")]:
        ev, ct = d[d["is_event"] == 1], d[d["is_event"] == 0]
        a, bb = int(ev[col].sum()), int(ct[col].sum())
        o, lo, hi = E.woolf(a, len(ev), bb, len(ct))
        rows.append(dict(가설=name, 이벤트비율=round(a / len(ev), 3), 대조비율=round(bb / len(ct), 3), OR=o, CI_low=lo, CI_high=hi, 이벤트건수=a))
    ev_tab = pd.DataFrame(rows).set_index("가설")
    print(f"\n[3] 자차 급감속 이벤트 {int(d['is_event'].sum())}건 vs 대조 {int((d['is_event'] == 0).sum())}개\n", ev_tab.to_string())
    ev_tab.to_csv(tdir / "ego_hypotheses_events.csv")


if __name__ == "__main__":
    main()

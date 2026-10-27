"""자차(AV) 전체 주행의 가속도·저크(Jerk) 분포와 '정지 방식' 비교.

- 자차 궤적은 측위 기반이라 다른 객체보다 노이즈가 작다(속도 필드와 위치 차분 속도의 차이 중앙값 약 0.04 m/s).
  그래서 자차만 가속도를 한 번 더 미분한 저크까지 계산한다(02와 같은 평활 후 미분, 저크는 0.3초 이동평균).
- 시나리오 앞뒤 약 1초(타임스텝 0~9, 100~109)는 자차 속도·위치가 왜곡돼 있어 빼고 10~99만 쓴다(10_ego_analysis.py 참고).
- 정지 에피소드: 3 m/s 이상 달리다 1 m/s 미만으로 떨어진 구간. 감속 시작 = 정지 직전 가속도가 -0.5 m/s^2 위였던 마지막 시점.
  최대 감속이 -3 m/s^2 이하면 '급정지', 그보다 완만하면 '일반 정지'로 나눠 같은 '정지' 상황끼리 비교한다.
  (급감속 vs 임의 주행 시점 비교보다 공정한 대조: 둘 다 결국 멈췄다)

실행: python src/13_ego_kinematics.py [--workers 4]
출력: outputs/ego_kinematics.parquet, outputs/ego_stops.parquet, outputs/tables/ego_jerk_summary.csv, ego_stop_compare.csv
"""
import argparse
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("c", "src/06_context_features.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

EGO, DT = "AV", 0.1
RUN_V, STOP_V, ONSET_ACC, HARD_ACC = 3.0, 1.0, -0.5, -3.0
EDGE_LO, EDGE_HI = 10, 99
JERK_COMFORT = 2.0       # 승차감 연구에서 흔히 쓰는 |jerk| 허용 범위 상한(m/s^3) 근사
COLS = c.COLS


def kin(df):
    g = df[(df["track_id"] == EGO) & df["timestep"].between(EDGE_LO, EDGE_HI)].sort_values("timestep")
    ts = g["timestep"].to_numpy()
    sp = np.hypot(g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy())
    s = pd.Series(sp)
    sm = s.rolling(5, center=True, min_periods=1).median().rolling(3, center=True, min_periods=1).mean().to_numpy()
    acc = np.gradient(sm, DT)
    jerk = pd.Series(np.gradient(acc, DT)).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return g, ts, sm, acc, jerk


def stops(ts, sm, acc, jerk):
    out = []
    i = 0
    n = len(sm)
    while i < n:
        if sm[i] >= RUN_V:
            j = i
            while j < n and sm[j] >= STOP_V:
                j += 1
            if j >= n:
                break
            k = j - 1
            while k > i and acc[k] <= ONSET_ACC:
                k -= 1
            on = k + 1
            if on <= j and on > 0:            # 감속 시작이 시나리오 첫 스텝이면 시작점을 모르므로 제외
                seg = slice(on, j + 1)
                out.append(dict(i_onset=on, i_stop=j, t_onset=int(ts[on]), t_stop=int(ts[j]),
                                v_onset=float(sm[on]), min_acc=float(acc[seg].min()),
                                min_jerk=float(jerk[seg].min()), max_abs_jerk=float(np.abs(jerk[seg]).max()),
                                brake_s=(j - on) * DT))
            while j < n and sm[j] < RUN_V:
                j += 1
            i = j
        else:
            i += 1
    return out


def work(path):
    df = pd.read_parquet(path, columns=COLS)
    sid, city = df["scenario_id"].iloc[0], df["city"].iloc[0]
    g, ts, sm, acc, jerk = kin(df)
    k = pd.DataFrame(dict(scenario_id=sid, city=city, timestep=ts.astype(np.int16), speed=sm.astype(np.float32),
                          acc=acc.astype(np.float32), jerk=jerk.astype(np.float32)))
    st = stops(ts, sm, acc, jerk)
    if st:
        mp = c.load_map(sid)
        by_t = {t: gg for t, gg in df.groupby("timestep")}
        x, y = g["position_x"].to_numpy(), g["position_y"].to_numpy()
        vx, vy = g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy()
        for s in st:
            j, on = s["i_stop"], s["i_onset"]
            h = max(j - 1, on)
            mf = c.map_feats(float(x[j]), float(y[j]), mp)
            sur_on = c.surroundings(by_t[int(ts[on])], EGO, float(x[on]), float(y[on]), float(vx[on]), float(vy[on]))
            sur_h = c.surroundings(by_t[int(ts[h])], EGO, float(x[h]), float(y[h]), float(vx[h]), float(vy[h]))
            s.update(scenario_id=sid, city=city, stop_dist_intersection=mf["dist_intersection"],
                     stop_on_intersection=mf["on_intersection"], stop_dist_crosswalk=mf["dist_crosswalk"],
                     lead_gap_stop=sur_h["lead_gap"], lead_gap_onset=sur_on["lead_gap"], lead_ttc_onset=sur_on["lead_ttc"],
                     lead_rel_onset=sur_on["lead_rel_speed"], n_agents_onset=sur_on["n_agents_30m"],
                     brake_dist=float(np.hypot(x[j] - x[on], y[j] - y[on])))
    return k, st


def woolf(a, n1, b, n0):
    A, B, C, D = a + .5, n1 - a + .5, b + .5, n0 - b + .5
    o = A * D / (B * C)
    se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
    return round(o, 2), round(np.exp(np.log(o) - 1.96 * se), 2), round(np.exp(np.log(o) + 1.96 * se), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = sorted(Path("data/val").glob("*/scenario_*.parquet"))
    if args.limit:
        files = files[:args.limit]
    ks, sts = [], []
    with Pool(args.workers) as pool:
        for i, (k, st) in enumerate(pool.imap_unordered(work, files, chunksize=50), 1):
            ks.append(k)
            sts += st
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    K = pd.concat(ks, ignore_index=True)
    S = pd.DataFrame(sts)
    K.to_parquet("outputs/ego_kinematics.parquet", index=False)
    S.to_parquet("outputs/ego_stops.parquet", index=False)
    tdir = Path("outputs/tables")

    mv = K[K["speed"] >= 1.0]
    q = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    summ = pd.DataFrame({"가속도(m/s^2)": mv["acc"].quantile(q).values, "저크(m/s^3)": mv["jerk"].quantile(q).values},
                        index=[f"{p*100:g}%" for p in q]).round(2)
    print(f"[1] 자차 주행 {K['scenario_id'].nunique():,}개 시나리오, 이동 중(>=1 m/s) {len(mv):,}스텝 ({len(mv)*DT/3600:.1f}시간)")
    print(summ.to_string())
    print(f"    |저크| > {JERK_COMFORT} 비율: {(mv['jerk'].abs() > JERK_COMFORT).mean():.3%},  가속도 <= -3 비율: {(mv['acc'] <= -3).mean():.3%}")
    summ.to_csv(tdir / "ego_jerk_summary.csv")

    S["hard"] = S["min_acc"] <= HARD_ACC
    S["near_inter8"] = S["stop_dist_intersection"] <= 8
    S["near_cross8"] = S["stop_dist_crosswalk"] <= 8
    S["lead_behind15"] = S["lead_gap_stop"] <= 15
    S["lead_at_onset"] = S["lead_gap_onset"].notna()
    S["ttc_onset4"] = S["lead_ttc_onset"] < 4
    S["vband"] = pd.cut(S["v_onset"], [0, 6, 9, 13, 100], labels=["<6", "6-9", "9-13", ">=13"])
    h, gs = S[S["hard"]], S[~S["hard"]]
    print(f"\n[2] 정지 에피소드 {len(S):,}건 (급정지 {len(h):,} / 일반 정지 {len(gs):,})")
    desc = S.groupby("hard").agg(건수=("hard", "size"), 감속시작속도=("v_onset", "median"), 제동시간_s=("brake_s", "median"),
                                 제동거리_m=("brake_dist", "median"), 최소가속도=("min_acc", "median"),
                                 최소저크=("min_jerk", "median")).round(2)
    desc.index = ["일반 정지", "급정지"][:len(desc)] if len(desc) == 2 else desc.index
    print(desc.to_string())
    rows = []
    for name, col in [("정지점이 교차로 8m 이내", "near_inter8"), ("정지점이 횡단보도 8m 이내", "near_cross8"),
                      ("선행 객체 뒤 정지(15m 이내)", "lead_behind15"), ("감속 시작 때 정면 선행 객체 있음", "lead_at_onset"),
                      ("감속 시작 때 선행 TTC<4초", "ttc_onset4")]:
        a, b = int(h[col].sum()), int(gs[col].sum())
        o, lo, hi = woolf(a, len(h), b, len(gs))
        rows.append(dict(조건=name, 급정지비율=round(a / len(h), 3), 일반정지비율=round(b / len(gs), 3), OR=o, CI_low=lo, CI_high=hi))
    cmp_ = pd.DataFrame(rows).set_index("조건")
    print("\n[3] 급정지 vs 일반 정지 (같은 '정지' 상황끼리 비교, Woolf OR)\n", cmp_.to_string())
    cmp_.to_csv(tdir / "ego_stop_compare.csv")
    print("\n[4] 감속 시작 속도 구간별 급정지 비율 / 교차로 정지 비율")
    print(S.groupby("vband", observed=True).agg(n=("hard", "size"), 급정지비율=("hard", "mean"),
                                                교차로8m=("near_inter8", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()

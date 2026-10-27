"""정지로 끝나는 자차 급감속 이벤트의 정지 지점 분석.
정지 지점이 (a) 선행 객체 뒤(15m 이내)인지, (b) 교차로 경계 바로 앞(8m 이내)인지를 자차 대조의 기저율과 비교한다.
주의: AV2 지도에는 신호등 상태와 정지선 정보가 없어서 '교차로 경계 앞 정지'는 정지선·신호 정지의 근사 지표일 뿐이다.

실행: python src/12_stop_points.py
출력: outputs/tables/ego_stop_points.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("e", "src/10_ego_analysis.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
c = e.c
EGO = "AV"
STOP_V = 1.0
NEAR_INTER, NEAR_LEAD = 8.0, 15.0


def stop_point(ts, sp, t0, horizon=30):
    """t0 이후 3초 내 첫 정지 시점과, 정지 직전 마지막 주행(>=1 m/s) 시점의 인덱스."""
    idx = np.where((ts >= t0) & (ts <= t0 + horizon))[0]
    stops = [i for i in idx if sp[i] < STOP_V]
    if not stops:
        return None
    mv = [i for i in idx if i < stops[0] and sp[i] >= STOP_V]
    return stops[0], (mv[-1] if mv else stops[0])


def analyze_scenario(sid, df, t0s, mp):
    by_t = {t: g for t, g in df.groupby("timestep")}
    g = df[df["track_id"] == EGO].sort_values("timestep")
    ts, x, y = g["timestep"].to_numpy(), g["position_x"].to_numpy(), g["position_y"].to_numpy()
    vx, vy = g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy()
    sp = np.hypot(vx, vy)
    out = []
    for t0 in t0s:
        sp_ = stop_point(ts, sp, t0)
        if sp_ is None:
            continue
        i_stop, i_h = sp_
        mf = c.map_feats(float(x[i_stop]), float(y[i_stop]), mp)
        s = c.surroundings(by_t[int(ts[i_h])], EGO, float(x[i_h]), float(y[i_h]), float(vx[i_h]), float(vy[i_h]))
        out.append(dict(scenario_id=sid, t=int(t0), t_stop=int(ts[i_stop]), stop_dist_intersection=mf["dist_intersection"],
                        stop_dist_crosswalk=mf["dist_crosswalk"], stop_on_intersection=mf["on_intersection"], lead_gap=s["lead_gap"]))
    return out


def summarize(stops, ctrl):
    stops = stops.copy()
    lead = (stops["lead_gap"] <= NEAR_LEAD).to_numpy()
    inter = (stops["stop_dist_intersection"] <= NEAR_INTER).to_numpy()
    stops["유형"] = np.select([lead, inter], ["선행 객체 뒤 정지(15m 이내)", "교차로 경계 앞 정지(8m 이내, 정지선·신호 근사)"], "기타")
    comp = stops["유형"].value_counts().to_frame("건수").assign(비율=lambda d: (d["건수"] / len(stops)).round(3))
    base = (ctrl["dist_intersection"] <= NEAR_INTER).mean()
    a, n = int(inter.sum()), len(stops)
    b, n0 = int((ctrl["dist_intersection"] <= NEAR_INTER).sum()), len(ctrl)   # 지도에 교차로가 없는 경우(NaN)는 '근접 아님'으로 센다
    o, lo, hi = e.woolf(a, n, b, n0)
    return stops, comp, dict(stop_share=round(a / n, 3), ctrl_share=round(base, 3), OR=o, lo=lo, hi=hi)


def main():
    d = pd.read_parquet("outputs/ego_samples.parquet")
    ev, ct = d[d["is_event"] == 1], d[d["is_event"] == 0]
    cat = e.classify(ev)[["scenario_id", "t", "cat"]]
    rows = []
    for sid, g in ev.groupby("scenario_id"):
        df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=c.COLS)
        rows += analyze_scenario(sid, df, list(g["t"]), c.load_map(sid))
    stops = pd.DataFrame(rows).merge(cat, on=["scenario_id", "t"], how="left")
    stops, comp, cmp_ = summarize(stops, ct)
    print(f"정지로 끝나는 자차 이벤트 {len(stops)}건 (전체 {len(ev)}건 중)\n")
    print("[1] 정지 상황 구성\n", comp)
    print("\n[2] 정지 지점이 교차로 경계 8m 이내인 비율: 이벤트 %.1f%% vs 자차 대조(주행 중 임의 시점) %.1f%%  (OR %.2f, 95%% CI %.2f~%.2f)"
          % (cmp_["stop_share"] * 100, cmp_["ctrl_share"] * 100, cmp_["OR"], cmp_["lo"], cmp_["hi"]))
    print("\n[3] 분류별 정지 상황\n", pd.crosstab(stops["cat"], stops["유형"]))
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    stops.round(2).to_csv("outputs/tables/ego_stop_points.csv", index=False)


if __name__ == "__main__":
    main()

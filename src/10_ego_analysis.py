"""자차(AV) 급감속 분석: 설명 가능/불가 분류, 자차 전용 대조군 비교, 임계값 민감도.

- 이벤트 시작 시점 -1.0초 ~ +0.5초 구간에서 선행 객체의 최소 TTC를 구한다(min_ttc_win).
- 분류(우선순위 순): 선행 객체(TTC) > 교차로·횡단보도 근접 > 보행자·자전거 근접 > 설명 불가
- 경계 제외: 자차 궤적은 시나리오 앞뒤 약 1초(타임스텝 0~9, 100~109)에서 속도·위치가 왜곡돼 있어
  (속도 필드 0~1스텝 값이 항상 같고, 위치 차분 속도가 경계에서 비현실적으로 오르내림) 이 구간에 걸친 이벤트·대조는 뺀다.
- 대조군: 같은 시나리오에서 자차가 급감속 후보 이벤트 앞뒤 3초 밖에서 3.2 m/s 이상으로 달린 시점(이벤트당 5개)
  -> 분류 규칙이 대조군에서도 똑같이 '설명 가능'으로 나오면 그 규칙은 설명력이 없다.

실행: python src/10_ego_analysis.py
출력: outputs/ego_samples.parquet, outputs/tables/ego_*.csv
"""
import importlib.util
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("c", "src/06_context_features.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

EGO = "AV"
WIN_PRE, WIN_POST = 10, 5
CTRL_PER_EVENT = 5
EDGE_LO, EDGE_HI = 10, 99   # 자차 경계 왜곡 구간 제외
VRU = {"pedestrian", "cyclist"}


def window_feats(by_t, ego, t0):
    idx = ego.set_index("timestep")
    ttcs, gaps = [], []
    for t in range(t0 - WIN_PRE, t0 + WIN_POST + 1):
        g_t = by_t.get(t)
        if g_t is None or t not in idx.index:
            continue
        r = idx.loc[t]
        s = c.surroundings(g_t, EGO, float(r["position_x"]), float(r["position_y"]),
                           float(r["velocity_x"]), float(r["velocity_y"]))
        if not np.isnan(s["lead_ttc"]):
            ttcs.append(s["lead_ttc"])
        if not np.isnan(s["lead_gap"]):
            gaps.append(s["lead_gap"])
    return dict(min_ttc_win=min(ttcs) if ttcs else np.nan, min_gap_win=min(gaps) if gaps else np.nan)


def vru_near(g_t, x, y, r=20.0):
    o = g_t[g_t["object_type"].isin(VRU)]
    return 0 if o.empty else int((np.hypot(o["position_x"].to_numpy() - x, o["position_y"].to_numpy() - y) <= r).sum())


def ego_rows_df(sid, df, ev_starts, windows, mp, n_ctrl):
    by_t = {t: g for t, g in df.groupby("timestep")}
    tracks = {tid: g.sort_values("timestep") for tid, g in df.groupby("track_id")}
    ego = tracks[EGO]
    rows = []

    def add(t, is_event):
        r = c.make_row(sid, by_t, tracks, EGO, int(t), is_event, mp)
        if r is None:
            return
        g_t = by_t[int(t)]
        e = g_t[g_t["track_id"] == EGO].iloc[0]
        r.update(window_feats(by_t, ego, int(t)))
        r["vru_20m"] = vru_near(g_t, float(e["position_x"]), float(e["position_y"]))
        rows.append(r)

    for t in ev_starts:
        add(t, True)
    ts = ego["timestep"].to_numpy()
    sp = np.hypot(ego["velocity_x"].to_numpy(), ego["velocity_y"].to_numpy())
    keep = (sp >= c.MIN_SPEED_CTRL) & (ts >= EDGE_LO) & (ts <= EDGE_HI)
    for s, e in windows:
        keep &= ~((ts >= s - c.EXCL) & (ts <= e + c.EXCL))
    cand = ts[keep]
    n = min(len(cand), n_ctrl)
    if n:
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        for t in rng.choice(cand, size=n, replace=False):
            add(t, False)
    return rows


def work(task):
    sid, starts, windows = task
    df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=c.COLS)
    return ego_rows_df(sid, df, starts, windows, c.load_map(sid), CTRL_PER_EVENT * len(starts))


def classify(d, ttc_th=4.0, near_m=15.0, vru_n=1):
    d = d.copy()
    d["f_lead"] = (d["min_ttc_win"] < ttc_th).to_numpy()
    d["f_map"] = ((d["dist_intersection"] <= near_m) | (d["dist_crosswalk"] <= near_m)).to_numpy()
    d["f_vru"] = (d["vru_20m"] >= vru_n).to_numpy()
    d["cat"] = np.select([d["f_lead"].to_numpy(), d["f_map"].to_numpy(), d["f_vru"].to_numpy()],
                         ["선행 객체(TTC)", "교차로·횡단보도 근접", "보행자·자전거 근접"], "설명 불가")
    return d


def woolf(a, n1, b, n0):
    A, B, C, D = a + .5, n1 - a + .5, b + .5, n0 - b + .5
    o = A * D / (B * C)
    se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
    return round(o, 2), round(np.exp(np.log(o) - 1.96 * se), 2), round(np.exp(np.log(o) + 1.96 * se), 2)


def report(d, rob, tdir):
    ev, ct = d[d["is_event"] == 1], d[d["is_event"] == 0]
    cl = classify(ev)
    print(f"자차 이벤트 {len(ev)}건 / 자차 대조 {len(ct)}개\n")
    print("[1] 설명 분류 (TTC<4s, 교차로·횡단보도 15m, 보행자·자전거 20m 이내)")
    print(cl["cat"].value_counts().to_frame("건수").assign(비율=lambda x: (x["건수"] / len(cl)).round(3)))

    sens = pd.DataFrame({f"근접 {m}m": [round((classify(ev, t, m)["cat"] == "설명 불가").mean(), 3) for t in [3, 4, 5, 6]]
                         for m in [10, 15, 25]}, index=[f"TTC<{t}s" for t in [3, 4, 5, 6]])
    print("\n[2] 임계값별 '설명 불가' 비율\n", sens)
    sens.to_csv(tdir / "ego_unexplained_sensitivity.csv")

    flags = {"선행 TTC<4s": lambda x: x["min_ttc_win"] < 4, "선행 15m 이내": lambda x: x["min_gap_win"] <= 15,
             "교차로 15m 이내": lambda x: x["dist_intersection"] <= 15, "횡단보도 15m 이내": lambda x: x["dist_crosswalk"] <= 15,
             "보행자·자전거 20m 이내": lambda x: x["vru_20m"] >= 1,
             "위 규칙 중 하나라도 해당(=설명 가능)": lambda x: classify(x)["cat"] != "설명 불가"}
    rows = []
    for name, f in flags.items():
        a, b = int(f(ev).sum()), int(f(ct).sum())
        o, lo, hi = woolf(a, len(ev), b, len(ct))
        rows.append(dict(규칙=name, 이벤트비율=round(a / len(ev), 3), 대조비율=round(b / len(ct), 3), OR=o, CI_low=lo, CI_high=hi))
    cmp_ = pd.DataFrame(rows).set_index("규칙")
    print("\n[3] 자차 이벤트 vs 자차 대조 (Woolf OR, 시나리오 매칭 미반영의 근사)\n", cmp_)
    cmp_.to_csv(tdir / "ego_vs_control.csv")

    un = cl[cl["cat"] == "설명 불가"].merge(
        rob[["scenario_id", "start_t", "min_acc", "speed_drop", "duration_s"]].rename(columns={"start_t": "t"}),
        on=["scenario_id", "t"], how="left")
    cols = ["scenario_id", "t", "speed_smooth", "min_acc", "speed_drop", "duration_s", "dist_intersection",
            "dist_crosswalk", "min_ttc_win", "min_gap_win", "vru_20m", "n_agents_30m"]
    un[cols].round(2).to_csv(tdir / "ego_unexplained_cases.csv", index=False)
    print(f"\n설명 불가 {len(un)}건 목록 저장: outputs/tables/ego_unexplained_cases.csv")


def main():
    rob = pd.read_parquet("outputs/events_robust.parquet")
    allev = pd.read_parquet("outputs/events.parquet", columns=["scenario_id", "track_id", "start_t", "end_t"])
    er, ea = rob[rob["track_id"] == EGO], allev[allev["track_id"] == EGO]
    n0 = len(er)
    er = er[(er["start_t"] >= EDGE_LO) & (er["end_t"] <= EDGE_HI)]
    print(f"경계(0~9, 100~109 스텝)에 걸친 자차 이벤트 {n0 - len(er)}건 제외")
    win = {s: list(zip(g["start_t"], g["end_t"])) for s, g in ea.groupby("scenario_id")}
    tasks = [(s, list(g["start_t"]), win.get(s, [])) for s, g in er.groupby("scenario_id")]
    print(f"자차 이벤트 {len(er)}건, 시나리오 {len(tasks)}개 처리")
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=10):
            rows += r
    d = pd.DataFrame(rows)
    d.to_parquet("outputs/ego_samples.parquet", index=False)
    tdir = Path("outputs/tables")
    tdir.mkdir(parents=True, exist_ok=True)
    report(d, rob, tdir)


if __name__ == "__main__":
    main()

"""자차(ego, track_id == "AV") 분석 가능성 점검.
(1) 원본에서 AV 트랙의 형태 확인, (2) 자차 이벤트 수/발생률/glitch 비율을 타 객체와 비교,
(3) 자차 급감속 시작 시점의 선행 객체·TTC 분포 미리보기.
자차는 시나리오 앞뒤 약 1초(타임스텝 0~9, 100~109)가 왜곡돼 있어 이 구간에 걸친 이벤트·표본을 빼고, 노출 시간도 90스텝 기준으로 센다.

실행: python src/09_ego_check.py
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.1
EGO = "AV"
EDGE_LO, EDGE_HI = 10, 99   # 10_ego_analysis.py와 같은 기준


def inside(ev):
    """경계 왜곡 구간에 걸치지 않은 이벤트만."""
    return ev[(ev["start_t"] >= EDGE_LO) & (ev["end_t"] <= EDGE_HI)]


def raw_check(n=300):
    files = sorted(Path("data/val").glob("*/scenario_*.parquet"))[:n]
    has, types, cats, steps = 0, Counter(), Counter(), []
    for f in files:
        d = pd.read_parquet(f, columns=["track_id", "object_type", "object_category", "timestep"])
        a = d[d["track_id"] == EGO]
        if len(a):
            has += 1
            types.update(a["object_type"].unique())
            cats.update(a["object_category"].unique())
            steps.append(len(a))
    print(f"[1] 원본 표본 {len(files)}개 시나리오 중 AV 트랙 존재: {has}개")
    print(f"    object_type: {dict(types)} / object_category: {dict(cats)} / 트랙 길이 중앙값: {np.median(steps) if steps else 'NA'}")


def summarize(ev_all, ev_rob, tr, ctx):
    ego_tr, oth_tr = tr[(tr["track_id"] == EGO)], tr[(tr["track_id"] != EGO) & tr["eligible"]]
    h_e = len(ego_tr) * (EDGE_HI - EDGE_LO + 1) * DT / 3600
    h_o = oth_tr["n_steps"].sum() * DT / 3600
    e_all0, o_all = ev_all[ev_all["track_id"] == EGO], ev_all[ev_all["track_id"] != EGO]
    e_rob0, o_rob = ev_rob[ev_rob["track_id"] == EGO], ev_rob[ev_rob["track_id"] != EGO]
    e_all, e_rob = inside(e_all0), inside(e_rob0)
    print(f"\n경계 제외: 자차 후보 이벤트 {len(e_all0)} -> {len(e_all)}, robust {len(e_rob0)} -> {len(e_rob)}")
    print(f"\n[2] 자차 vs 그 외 (분석 대상 유형)")
    print(pd.DataFrame({
        "트랙수": [len(ego_tr), len(oth_tr)], "노출_객체시간": [round(h_e, 1), round(h_o, 1)],
        "후보 이벤트": [len(e_all), len(o_all)], "glitch 비율": [round(e_all["glitch"].mean(), 3) if len(e_all) else np.nan,
                                                              round(o_all["glitch"].mean(), 3)],
        "robust 이벤트": [len(e_rob), len(o_rob)],
        "robust 건/객체시간": [round(len(e_rob) / h_e, 2) if h_e else np.nan, round(len(o_rob) / h_o, 2)],
    }, index=["자차(AV)", "그 외"]).T)

    keep = set(zip(e_rob["scenario_id"], e_rob["start_t"]))   # 시작·끝 모두 경계 밖인 이벤트만
    ce = ctx[(ctx["track_id"] == EGO) & (ctx["is_event"] == 1)]
    ce = ce[[k in keep for k in zip(ce["scenario_id"], ce["t"])]]
    print(f"\n[3] 자차 robust 급감속 시작 시점의 맥락 (n={len(ce)})")
    if len(ce):
        ttc = ce["lead_ttc"].dropna()
        print(pd.Series({
            "선행 객체 있음 비율": round(ce["lead_gap"].notna().mean(), 3),
            "선행 15m 이내 비율": round((ce["lead_gap"] <= 15).mean(), 3),
            "TTC 산출 가능(접근 중) 비율": round(ce["lead_ttc"].notna().mean(), 3),
            "TTC 중앙값(초)": round(ttc.median(), 1) if len(ttc) else np.nan,
            "TTC < 4초 비율(전체 대비)": round((ce["lead_ttc"] < 4).mean(), 3),
            "교차로 15m 이내 비율": round((ce["dist_intersection"] <= 15).mean(), 3),
            "횡단보도 15m 이내 비율": round((ce["dist_crosswalk"] <= 15).mean(), 3),
            "직전 급가속(>4) 비율": round(((ce["pre_max_acc"] > 4) & (ce["seg_pos"] >= 15)).mean(), 3),
        }).to_string())
    return e_rob


def main():
    raw_check()
    ev_all = pd.read_parquet("outputs/events.parquet")
    ev_rob = pd.read_parquet("outputs/events_robust.parquet")
    tr = pd.read_parquet("outputs/tracks.parquet")
    ctx = pd.read_parquet("outputs/context_samples.parquet")
    summarize(ev_all, ev_rob, tr, ctx)


if __name__ == "__main__":
    main()

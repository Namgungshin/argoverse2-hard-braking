"""D5 본 분석: 이벤트 vs 대조 맥락 비교.
(1) 속도 구간별 기술 비교, (2) 로지스틱 회귀(시나리오 클러스터 강건 SE)
- 대조군을 이벤트당 2개 뽑은 case-control 설계라 OR은 '해당 조건일 때 급감속 시작의 상대적 가능성'으로 읽는다.
- 모델 A: 전체 샘플 / 모델 B: 트랙 시작 직후(seg_pos<15)와 직전 비현실 가속(>4 m/s^2) 샘플을 양쪽 다 제외

실행: python src/07_context_analysis.py
출력: outputs/tables/context_by_speed.csv, context_logit.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REF_CITY = "miami"
BASE_COLS = ["speed5", "on_inter", "near_inter15", "near_cross15", "has_lead", "lead_close",
             "agents5", "type_bus", "type_cyclist", "type_motorcyclist"]
LABELS = {"speed5": "속도 (+5 m/s)", "on_inter": "교차로 차선 위", "near_inter15": "교차로 15m 이내",
          "near_cross15": "횡단보도 15m 이내", "has_lead": "정면 선행 객체 있음",
          "lead_close": "선행 객체 15m 이내", "agents5": "주변 객체 수 (+5명)",
          "type_bus": "유형: bus (vs vehicle)", "type_cyclist": "유형: cyclist", "type_motorcyclist": "유형: motorcyclist"}


def prepare(df):
    d = df.dropna(subset=["speed_smooth"]).copy()
    d["speed5"] = d["speed_smooth"] / 5
    d["on_inter"] = d["on_intersection"].fillna(0)
    d["near_inter15"] = (d["dist_intersection"] <= 15).astype(float)
    d["near_cross15"] = (d["dist_crosswalk"] <= 15).astype(float)
    d["has_lead"] = d["lead_gap"].notna().astype(float)
    d["lead_close"] = (d["lead_gap"] <= 15).astype(float)
    d["agents5"] = d["n_agents_30m"] / 5
    d["early"] = (d["seg_pos"] < 15).astype(float)
    d["pulse"] = ((d["pre_max_acc"] > 4) & (d["seg_pos"] >= 15)).astype(float)
    for t in ["bus", "cyclist", "motorcyclist"]:
        d[f"type_{t}"] = (d["object_type"] == t).astype(float)
    for c in sorted(d["city"].unique()):
        if c != REF_CITY:
            d[f"city_{c}"] = (d["city"] == c).astype(float)
    return d


def logit(X, y, clusters, iters=50):
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(k)
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(k))
    S = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(clusters)).sum().to_numpy()
    G = len(S)
    cov = Hinv @ (S.T @ S) @ Hinv * (G / max(G - 1, 1))
    return b, np.sqrt(np.diag(cov))


def run(sub):
    cities = [c for c in sub.columns if c.startswith("city_")]
    cols = BASE_COLS + cities
    X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy(float) for c in cols])
    b, se = logit(X, sub["is_event"].to_numpy(float), sub["scenario_id"].to_numpy())
    res = pd.DataFrame({"OR": np.exp(b[1:]), "CI_low": np.exp(b[1:] - 1.96 * se[1:]),
                        "CI_high": np.exp(b[1:] + 1.96 * se[1:])}, index=cols).round(2)
    res.index = [LABELS.get(c, c.replace("city_", "도시: ") + f" (vs {REF_CITY})") for c in cols]
    return res


def main():
    d = prepare(pd.read_parquet("outputs/context_samples.parquet"))
    tdir = Path("outputs/tables")
    tdir.mkdir(parents=True, exist_ok=True)

    print(f"샘플 {len(d):,}개 (이벤트 {int(d['is_event'].sum()):,})")
    print("\n[1] 트랙 시작 직후 / 직전 비현실 가속 비율 (이벤트 vs 대조)")
    print(d.groupby("is_event")[["early", "pulse"]].mean().round(3))

    d["band"] = pd.cut(d["speed_smooth"], [0, 6, 9, 13, 200], labels=["<6", "6-9", "9-13", ">=13"])
    desc = d.groupby(["band", "is_event"], observed=True).agg(
        n=("t", "size"), 교차로위=("on_inter", "mean"), 교차로15m=("near_inter15", "mean"),
        횡단보도15m=("near_cross15", "mean"), 선행있음=("has_lead", "mean"),
        선행거리중앙값=("lead_gap", "median"), 주변객체중앙값=("n_agents_30m", "median"),
        pulse=("pulse", "mean"), early=("early", "mean")).round(3)
    print("\n[2] 속도 구간별 비교 (is_event: 1=급감속 시작, 0=대조)\n", desc.to_string())
    desc.to_csv(tdir / "context_by_speed.csv")

    a = run(d)
    clean = d[(d["early"] == 0) & (d["pulse"] == 0)]
    b = run(clean)
    out = a.join(b, lsuffix="_A전체", rsuffix="_B정제")
    print(f"\n[3] 로지스틱 회귀 OR (95% CI, 시나리오 클러스터 강건)  A: 전체 {len(d):,}개 / B: 정제 {len(clean):,}개 "
          f"(이벤트 {int(clean['is_event'].sum()):,})")
    print(out.to_string())
    out.to_csv(tdir / "context_logit.csv")


if __name__ == "__main__":
    main()

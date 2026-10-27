"""급감속 이벤트 분석: 필터 깔때기, 민감도, 도시/유형별 발생률(+부트스트랩 CI), 개요 그림.

실행: python src/04_analysis.py
출력: outputs/tables/*.csv, outputs/figures/overview.png, outputs/events_robust.parquet
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DT = 0.1
ACC_FLOOR = -9.8   # 물리적 제동 한계 (약 1g)
MIN_DROP = 3.0     # robust 정의: 속도 감소 하한 (m/s, 약 11 km/h)
MIN_DUR = 0.7      # robust 정의: 최소 지속 시간 (초). 샘플 검토에서 0.5초대 펄스가 노이즈로 확인됨
N_BOOT = 300


def hours(tr):
    return tr["n_steps"].sum() * DT / 3600


def funnel(ev):
    e1 = ev[~ev["glitch"]]
    e2 = e1[e1["min_acc"] >= ACC_FLOOR]
    e3 = e2[e2["speed_drop"] >= MIN_DROP]
    e4 = e3[e3["duration_s"] >= MIN_DUR - 1e-6]
    steps = [("추출 전체 (<= -3 m/s^2, >= 0.5초)", ev), ("glitch 제외", e1),
             (f"물리 한계 내 (min_acc >= {ACC_FLOOR})", e2),
             (f"속도 감소 >= {MIN_DROP} m/s", e3),
             (f"지속 >= {MIN_DUR}초 (robust)", e4)]
    rows, prev = [], len(ev)
    for name, d in steps:
        rows.append(dict(단계=name, 건수=len(d), 직전대비=round(len(d) / prev, 3) if prev else np.nan))
        prev = len(d)
    return pd.DataFrame(rows), e2, e3, e4


def sensitivity(e_plausible, h):
    """mean_acc(이벤트 평균 가속도) 임계값 x 속도 감소 하한 그리드. 값: 건수, 건/객체-시간."""
    accs, drops = [-3, -4, -5, -6], [2, 3, 4, 5]
    rate = pd.DataFrame(index=[f"mean_acc<={a}" for a in accs],
                        columns=[f"drop>={d}" for d in drops], dtype=float)
    cnt = rate.copy()
    for a in accs:
        for d in drops:
            n = ((e_plausible["mean_acc"] <= a) & (e_plausible["speed_drop"] >= d)).sum()
            cnt.loc[f"mean_acc<={a}", f"drop>={d}"] = n
            rate.loc[f"mean_acc<={a}", f"drop>={d}"] = round(n / h, 2)
    return cnt.astype(int), rate


def duration_diagnostic(e3, h):
    """지속 시간 하한별 건수/발생률/상한 근처(min_acc < -9) 비율. 상한 근처 비율이 급락하면 짧은 펄스가 노이즈였다는 증거."""
    rows = []
    for d in [0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
        x = e3[e3["duration_s"] >= d - 1e-6]
        rows.append(dict(최소지속_초=d, 건수=len(x), 건_per_객체시간=round(len(x) / h, 2),
                         상한근처_비율=round((x["min_acc"] < -9).mean(), 3) if len(x) else np.nan,
                         speed_start_중앙값=round(x["speed_start"].median(), 1) if len(x) else np.nan))
    return pd.DataFrame(rows)


def rate_ci(e, tr, col, n_boot=N_BOOT, seed=0):
    """시나리오 단위 부트스트랩 95% 신뢰구간 (건/객체-시간)."""
    el = tr[tr["eligible"]]
    ex = el.groupby([col, "scenario_id"])["n_steps"].sum().mul(DT / 3600).rename("h")
    ct = e.groupby([col, "scenario_id"]).size().rename("n")
    d = pd.concat([ex, ct], axis=1).fillna(0)
    rng = np.random.default_rng(seed)
    rows = {}
    for key, g in d.groupby(level=0):
        h, c, m = g["h"].to_numpy(), g["n"].to_numpy(), len(g)
        idx = rng.integers(0, m, size=(n_boot, m))
        r = c[idx].sum(1) / h[idx].sum(1)
        rows[key] = (np.percentile(r, 2.5), np.percentile(r, 97.5))
    return pd.DataFrame(rows, index=["ci_low", "ci_high"]).T.round(2)


def rate_by(e, tr, col):
    el = tr[tr["eligible"]]
    h = el.groupby(col)["n_steps"].sum() * DT / 3600
    n = e.groupby(col).size().reindex(h.index).fillna(0)
    out = pd.DataFrame({"객체수": el.groupby(col).size(), "노출_객체시간": h.round(1),
                        "이벤트": n.astype(int), "건_per_객체시간": (n / h).round(2)})
    out = out.join(rate_ci(e, tr, col))
    return out.sort_values("건_per_객체시간", ascending=False)


def _bar(ax, df, color, title):
    d = df.sort_values("건_per_객체시간")
    v = d["건_per_객체시간"].to_numpy()
    xerr = [np.clip(v - d["ci_low"].to_numpy(), 0, None), np.clip(d["ci_high"].to_numpy() - v, 0, None)]
    ax.barh(d.index.astype(str), v, xerr=xerr, color=color, capsize=3)
    ax.set_title(title)


def make_figure(rob, city_rate, type_rate, path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    _bar(ax[0, 0], city_rate, "#4c72b0", "도시별 급감속 발생률 (건/객체-시간, 95% CI)")
    _bar(ax[0, 1], type_rate, "#dd8452", "객체 유형별 발생률 (건/객체-시간, 95% CI)")
    ax[1, 0].hist(rob["min_acc"], bins=40, color="#55a868")
    ax[1, 0].set_title("robust 이벤트의 최대 감속도 분포")
    ax[1, 0].set_xlabel("min_acc (m/s^2)")
    ax[1, 1].hist(rob["start_t"], bins=np.arange(-0.5, 106.5, 1), color="#c44e52")  # 정수 1개당 1구간
    ax[1, 1].axvline(49.5, color="k", ls="--", lw=1)
    ax[1, 1].set_title("이벤트 시작 timestep (점선=관측/예측 경계, 노출 비보정)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    ev = pd.read_parquet("outputs/events.parquet")
    tr = pd.read_parquet("outputs/tracks.parquet")
    el = tr[tr["eligible"]]
    h = hours(el)
    tdir = Path("outputs/tables")
    tdir.mkdir(parents=True, exist_ok=True)

    f, e_plaus, e_drop, rob = funnel(ev)
    print(f"분석 대상 노출 시간: {h:,.0f} 객체-시간\n")
    print("[1] 필터 깔때기\n", f.to_string(index=False), "\n")
    f.to_csv(tdir / "funnel.csv", index=False)

    cnt, rate = sensitivity(e_plaus, h)
    print("[2] 민감도: 건수\n", cnt, "\n\n    민감도: 건/객체-시간\n", rate, "\n")
    cnt.to_csv(tdir / "sensitivity_count.csv")
    rate.to_csv(tdir / "sensitivity_rate.csv")

    dd = duration_diagnostic(e_drop, h)
    print("[2b] 지속 시간 하한별 진단 (glitch 제외, 물리 한계 내, 속도 감소 조건 충족 이벤트)\n", dd.to_string(index=False), "\n")
    dd.to_csv(tdir / "duration_diagnostic.csv", index=False)

    by_type = rate_by(rob, tr, "object_type")
    by_city = rate_by(rob, tr, "city")
    print("[3] 객체 유형별 (robust)\n", by_type, "\n")
    print("[4] 도시별 (robust)\n", by_city, "\n")
    by_type.to_csv(tdir / "rate_by_type.csv")
    by_city.to_csv(tdir / "rate_by_city.csv")

    print("[5] robust 이벤트 요약")
    print(rob[["duration_s", "min_acc", "speed_start", "speed_drop"]].describe().round(2))
    print(f"\n관측 구간에서 시작: {rob['started_in_observed'].mean():.1%}, "
          f"focal 객체: {rob['is_focal'].mean():.1%}, severe(<=-4.5): {rob['severe'].mean():.1%}")

    make_figure(rob, by_city, by_type, Path("outputs/figures/overview.png"))
    rob.to_parquet("outputs/events_robust.parquet", index=False)
    print("\n저장: outputs/tables/*.csv, outputs/figures/overview.png, outputs/events_robust.parquet")


if __name__ == "__main__":
    main()

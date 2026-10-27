"""자차 급감속 원인 추적: (1) 3초 내 정지·30도 이상 회전 비교, (2) 사례 시각화(설명 불가 3 + TTC 설명 3, 무작위 추출).

실행: python src/11_ego_cases.py [seed]
출력: outputs/figures/ego_cases.png, outputs/tables/ego_future_flags.csv, 콘솔 요약
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

spec = importlib.util.spec_from_file_location("e", "src/10_ego_analysis.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
c = e.c
EGO = "AV"
STATIC = {"background", "static", "construction", "unknown"}
MARK = {"pedestrian": "^", "cyclist": "v", "motorcyclist": "D", "bus": "s"}


def ego_arrays(df):
    g = df[df["track_id"] == EGO].sort_values("timestep")
    return (g["timestep"].to_numpy(), g["position_x"].to_numpy(), g["position_y"].to_numpy(),
            g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy())


def future_flags(ts, vx, vy, t0, horizon=30):
    """t0 이후 3초 내 정지(속도<1 m/s) 여부, 진행 방향이 30도 넘게 바뀌었는지(회전)."""
    m = (ts >= t0) & (ts <= t0 + horizon)
    if m.sum() < 10:
        return np.nan, np.nan
    ux, uy = vx[m], vy[m]
    sp = np.hypot(ux, uy)
    moving = np.where(sp > 1.0)[0]
    if len(moving) == 0:
        return 1.0, np.nan
    a0, a1 = np.arctan2(uy[moving[0]], ux[moving[0]]), np.arctan2(uy[moving[-1]], ux[moving[-1]])
    d = np.degrees(abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi))
    return float(sp.min() < 1.0), float(d > 30)


def load_lanes(sid):
    m = json.load(open(c.ROOT / sid / f"log_map_archive_{sid}.json"))
    lanes = [(np.array([[p["x"], p["y"]] for p in l["centerline"]]), bool(l["is_intersection"]))
             for l in m["lane_segments"].values()]
    cw = []
    for pc in m["pedestrian_crossings"].values():
        a, b = pc["edge1"], pc["edge2"]
        cw.append(np.array([[a[0]["x"], a[0]["y"]], [a[1]["x"], a[1]["y"]], [b[1]["x"], b[1]["y"]], [b[0]["x"], b[0]["y"]]]))
    return lanes, cw


def draw_case(ax, df, lanes, cw, t0, title, R=45):
    ts, x, y, vx, vy = ego_arrays(df)
    i0 = int(np.where(ts == t0)[0][0])
    cx, cy = x[i0], y[i0]
    for pts, inter in lanes:
        if np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).min() <= R + 30:
            ax.plot(pts[:, 0], pts[:, 1], color="#e08a1e" if inter else "#bbbbbb", lw=1.0, zorder=1)
    for poly in cw:
        if np.hypot(poly[:, 0] - cx, poly[:, 1] - cy).min() <= R + 30:
            ax.add_patch(Polygon(poly, closed=True, fc="#9ecae1", ec="#3182bd", alpha=0.6, zorder=2))
    win = (ts >= t0 - 30) & (ts <= t0 + 30)
    sc = ax.scatter(x[win], y[win], c=np.hypot(vx, vy)[win], cmap="viridis", s=14, vmin=0, vmax=15, zorder=4)
    ax.scatter([cx], [cy], marker="*", s=220, color="red", zorder=7)
    for tid, g in df[df["track_id"] != EGO].groupby("track_id"):
        g = g.sort_values("timestep")
        at = g[g["timestep"] == t0]
        if at.empty:
            continue
        px, py = float(at["position_x"].iloc[0]), float(at["position_y"].iloc[0])
        if np.hypot(px - cx, py - cy) > R:
            continue
        gg = g[(g["timestep"] >= t0 - 20) & (g["timestep"] <= t0 + 10)]
        typ = g["object_type"].iloc[0]
        ax.plot(gg["position_x"], gg["position_y"], color="k", lw=0.6, alpha=0.5, zorder=3)
        ax.scatter([px], [py], marker=MARK.get(typ, "o"), s=40, color="#999999" if typ in STATIC else "#d62728", zorder=6)
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = pd.read_parquet("outputs/ego_samples.parquet")
    rob = pd.read_parquet("outputs/events_robust.parquet")
    flags, cache = {}, {}
    for sid, g in d.groupby("scenario_id"):
        df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=c.COLS)
        cache[sid] = df
        ts, _, _, vx, vy = ego_arrays(df)
        for i, r in g.iterrows():
            flags[i] = future_flags(ts, vx, vy, int(r["t"]))
    d["stop3s"] = [flags[i][0] for i in d.index]
    d["turn30"] = [flags[i][1] for i in d.index]

    ev, ct = d[d["is_event"] == 1], d[d["is_event"] == 0]
    cl = e.classify(ev)
    groups = {"이벤트 전체": ev, "이벤트: 설명 불가": ev[cl["cat"] == "설명 불가"],
              "이벤트: TTC로 설명": ev[cl["cat"] == "선행 객체(TTC)"], "자차 대조": ct}
    tab = pd.DataFrame({k: dict(건수=len(v), **{"3초 내 정지": round(v["stop3s"].mean(), 3), "30도 이상 회전": round(v["turn30"].mean(), 3)})
                        for k, v in groups.items()}).T
    print("[1] 이벤트 시작 후 3초 동안의 자차 행동\n", tab)
    rows = []
    for name in ["3초 내 정지", "30도 이상 회전"]:
        col = "stop3s" if name.startswith("3초") else "turn30"
        for gname in ["이벤트 전체", "이벤트: 설명 불가"]:
            g = groups[gname]
            a, n1, b, n0 = int(g[col].sum()), int(g[col].notna().sum()), int(ct[col].sum()), int(ct[col].notna().sum())
            o, lo, hi = e.woolf(a, n1, b, n0)
            rows.append(dict(행동=name, 비교=f"{gname} vs 대조", OR=o, CI_low=lo, CI_high=hi))
    orr = pd.DataFrame(rows)
    print("\n[2] 오즈비 (Woolf 근사)\n", orr.to_string(index=False))
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.concat([tab.reset_index().rename(columns={"index": "집단"})]).to_csv("outputs/tables/ego_future_flags.csv", index=False)

    evm = cl.merge(rob[["scenario_id", "start_t", "min_acc", "speed_drop"]].rename(columns={"start_t": "t"}), on=["scenario_id", "t"], how="left")
    pick = pd.concat([evm[evm["cat"] == "설명 불가"].sample(min(3, int((evm["cat"] == "설명 불가").sum())), random_state=seed),
                      evm[evm["cat"] == "선행 객체(TTC)"].sample(min(3, int((evm["cat"] == "선행 객체(TTC)").sum())), random_state=seed)])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    sc = None
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        sid, t0 = r["scenario_id"], int(r["t"])
        lanes, cw = load_lanes(sid)
        ttl = (f"{r['cat']} | {sid[:8]} t={t0}\n속도 {r['speed_smooth']:.1f} m/s, 감속 {r['min_acc']:.1f} m/s², "
               f"TTC {r['min_ttc_win']:.1f}s" if not np.isnan(r["min_ttc_win"]) else
               f"{r['cat']} | {sid[:8]} t={t0}\n속도 {r['speed_smooth']:.1f} m/s, 감속 {r['min_acc']:.1f} m/s², 선행 없음/비접근")
        sc = draw_case(ax, cache[sid], lanes, cw, t0, ttl)
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.6, label="자차 속도 (m/s)")
    fig.suptitle("자차 급감속 사례 (위: 설명 불가, 아래: TTC로 설명 / 무작위 추출) — 빨간 별=급감속 시작, 주황=교차로 차선, 파랑=횡단보도, "
                 "빨간 점=동적 객체(검은 선=-2~+1초 궤적)", fontsize=10)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/figures/ego_cases.png", dpi=130, bbox_inches="tight")
    print("\n저장: outputs/figures/ego_cases.png (seed=%d)" % seed)


if __name__ == "__main__":
    main()

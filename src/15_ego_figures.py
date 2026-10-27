"""자차 분석 그림.
(1) ego_overview.png: 자차 가속도·저크 분포 + 급정지 vs 일반 정지 회귀 오즈비
(2) ego_case_<이름>.gif / .png: 대표 급정지 사례를 0.1초 단위로 재구성(지도 위 조감도 + 속도·가속도·저크 시계열)

실행: python src/15_ego_figures.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
spec = importlib.util.spec_from_file_location("k", "src/11_ego_cases.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
c = k.c
INK, MUTED, ACCENT, GRID = "#222222", "#777777", "#2a6fdb", "#e6e6e6"
FIG = Path("outputs/figures")
CASES = {  # (scenario_id, 이름, 설명)
    "crosswalk": ("dd332440-75b5-4cb9-8cb7-5d45acb5832c", "횡단보도 앞 급정지 (보행자 있음, 선행 차량 없음)"),
    "bus": ("bc529249-ea10-4dfd-a3fd-3f5879a2063c", "버스 인접 급정지 (교차로·횡단보도와 먼 단일로)"),
}
TYPE_COLOR = {"pedestrian": "#d62728", "cyclist": "#d62728", "bus": "#8c564b", "vehicle": "#555555", "motorcyclist": "#555555"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def overview():
    K = pd.read_parquet("outputs/ego_kinematics.parquet")
    mv = K[K["speed"] >= 1.0]
    lg = pd.read_csv("outputs/tables/ego_hypotheses_logit.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, col, rng, lines, unit, title in [
            (axes[0], "acc", (-6, 4), [-3], "m/s²", "자차 가속도 분포 (이동 중 42시간)"),
            (axes[1], "jerk", (-12, 12), [-2, 2], "m/s³", "자차 저크 분포")]:
        ax.hist(mv[col].clip(*rng), bins=120, range=rng, color=ACCENT, lw=0)
        ax.set_yscale("log")
        for v in lines:
            ax.axvline(v, color=INK, lw=1, ls="--")
        share = (mv[col] <= -3).mean() if col == "acc" else (mv[col].abs() > 2).mean()
        note = f"≤ -3 m/s²: {share:.2%}" if col == "acc" else f"|저크| > 2: {share:.1%}"
        ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=9, color=INK, va="top")
        ax.set_xlabel(unit, fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style(ax)
    ax = axes[2]
    lg = lg.iloc[::-1]
    y = np.arange(len(lg))
    ax.hlines(y, lg["CI_low"], lg["CI_high"], color=ACCENT, lw=2)
    ax.plot(lg["OR"], y, "o", color=ACCENT, ms=7, mec="white", mew=1.5)
    for yi, (o, lo, hi) in enumerate(zip(lg["OR"], lg["CI_low"], lg["CI_high"])):
        ax.text(max(hi, o) * 1.08, yi, f"{o:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, lg.index, fontsize=8.5, color=INK)
    ax.set_xlabel("오즈비 (로그 척도, 95% CI) — 1보다 크면 급정지 쪽", fontsize=9, color=MUTED)
    ax.set_title("급정지 vs 일반 정지 (정지 3,489건)", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "ego_overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def case(name, sid, desc):
    S = pd.read_parquet("outputs/ego_stops_feat.parquet")
    st = S[(S["scenario_id"] == sid) & (S["min_acc"] <= -3)].iloc[0]
    t_on, t_st = int(st["t_onset"]), int(st["t_stop"])
    df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=c.COLS)
    K = pd.read_parquet("outputs/ego_kinematics.parquet", filters=[("scenario_id", "==", sid)]).sort_values("timestep")
    lanes, cw = k.load_lanes(sid)
    ego = df[df["track_id"] == "AV"].set_index("timestep").sort_index()
    t0, t1 = max(t_on - 25, 10), min(t_st + 10, 99)
    cx, cy = ego.loc[t_on, ["position_x", "position_y"]].astype(float)
    R = 35

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_axes([0.03, 0.06, 0.48, 0.84])
    axs = [fig.add_axes([0.58, 0.68 - i * 0.29, 0.39, 0.22]) for i in range(3)]
    for pts, inter in lanes:
        if np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).min() <= R + 30:
            ax.plot(pts[:, 0], pts[:, 1], color="#e0a050" if inter else "#c8c8c8", lw=1, zorder=1)
    for poly in cw:
        if np.hypot(poly[:, 0] - cx, poly[:, 1] - cy).min() <= R + 30:
            ax.add_patch(Polygon(poly, closed=True, fc="#9ecae1", ec="#3182bd", alpha=0.6, zorder=2))
    e_tr = ego.loc[t0:t1]
    ax.plot(e_tr["position_x"], e_tr["position_y"], color=ACCENT, lw=1.5, alpha=0.4, zorder=3)
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("", fontsize=10, color=INK, loc="left")
    ego_dot = ax.scatter([], [], s=140, marker="s", color=ACCENT, zorder=6, label="자차(AV)")
    oth = ax.scatter([], [], s=30, zorder=5)
    ax.scatter([], [], s=30, color="#d62728", label="보행자·자전거")
    ax.scatter([], [], s=30, color="#555555", label="차량")
    ax.scatter([], [], s=30, color="#8c564b", label="버스")
    ax.add_patch(Polygon([[0, 0]], closed=True, fc="#9ecae1", ec="#3182bd", alpha=0.6, label="횡단보도"))
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    tsec = (K["timestep"].to_numpy() - t_on) / 10
    m = (K["timestep"] >= t0) & (K["timestep"] <= t1)
    cursors = []
    for a, col, unit, ref in [(axs[0], "speed", "속도 m/s", None), (axs[1], "acc", "가속도 m/s²", -3), (axs[2], "jerk", "저크 m/s³", -2)]:
        a.plot(tsec[m], K[col].to_numpy()[m], color=ACCENT, lw=2)
        if ref is not None:
            a.axhline(ref, color=INK, lw=1, ls="--")
        a.axvspan(0, (t_st - t_on) / 10, color="#f2f2f2", zorder=0)
        a.set_ylabel(unit, fontsize=8, color=MUTED)
        style(a)
        cursors.append(a.axvline(0, color="#d62728", lw=1.2))
    axs[2].set_xlabel("감속 시작 기준 시간 (초)  — 회색: 감속 시작~정지", fontsize=8, color=MUTED)
    fig.suptitle(f"{desc}  |  {sid[:8]}  최대 감속 {st['min_acc']:.1f} m/s², 최소 저크 {st['min_jerk']:.1f} m/s³",
                 fontsize=11, color=INK, x=0.03, ha="left")

    def frame(t):
        g = df[(df["timestep"] == t) & (df["track_id"] != "AV")]
        g = g[g["object_type"].isin(TYPE_COLOR) & (np.hypot(g["position_x"] - cx, g["position_y"] - cy) <= R + 5)]
        oth.set_offsets(g[["position_x", "position_y"]].to_numpy() if len(g) else np.empty((0, 2)))
        oth.set_color([TYPE_COLOR[o] for o in g["object_type"]])
        ego_dot.set_offsets([ego.loc[t, ["position_x", "position_y"]].astype(float).to_numpy()])
        title.set_text(f"t = {(t - t_on) / 10:+.1f}초")
        for cu in cursors:
            cu.set_xdata([(t - t_on) / 10] * 2)
        return [oth, ego_dot, title, *cursors]

    anim = FuncAnimation(fig, frame, frames=range(t0, t1 + 1), blit=False)
    anim.save(FIG / f"ego_case_{name}.gif", writer=PillowWriter(fps=8), dpi=80)
    frame(t_on)
    fig.savefig(FIG / f"ego_case_{name}.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    overview()
    for name, (sid, desc) in CASES.items():
        case(name, sid, desc)
    print("저장:", sorted(p.name for p in FIG.glob("ego_*")))


if __name__ == "__main__":
    main()

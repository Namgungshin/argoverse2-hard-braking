"""기획안의 남은 두 분석.

(A) 차선 경계 타입 (H1 보완): 감속 시작 시점 자차가 달리던 차선의 좌우 경계 표시
    - 자차 차선 = 가장 가까운 차량 차선 중심선 중 진행 방향이 자차와 45도 이내이고 3m 이내인 것
    - 경계 분류: 점선(DASHED_*, DOUBLE_DASH_*), 혼합(DASH_SOLID_*, SOLID_DASH_*), 실선(SOLID_*, DOUBLE_SOLID_*), 표시 없음(NONE)
    - '점선→실선 전환 구간': 지금 차선의 한쪽이 점선/혼합인데, 그 차선과 후속 차선(앞 40m 안)에서 같은 쪽이 실선으로 바뀌는 경우
      (교차로 진입 전 차선 변경 금지 구간이 시작되는 지점의 근사)
    - 비교: 급정지 vs 일반 정지(13번), 자차 급감속 이벤트 vs 자차 대조(10번)

(B) 객체 클래스별 저크: 정지 에피소드의 최소 저크·최소 가속도를, 감속 시작 때 30m 안에서 가장 가까운 동적 객체의 클래스로 나눠 비교
    - 다른 객체 자신의 저크는 트래킹 노이즈(2부 참고) 때문에 신뢰하기 어려워, 측위 기반으로 깨끗한 자차 저크만 쓴다.

실행: python src/16_lane_marks_class_jerk.py
출력: outputs/tables/lane_marks_stops.csv, lane_marks_events.csv, class_jerk.csv, outputs/figures/class_jerk_box.png
"""
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c = load("c", "src/06_context_features.py")
L = load("L", "src/07_context_analysis.py")
E = load("E", "src/10_ego_analysis.py")
EGO = "AV"
LANE_TYPES = {"VEHICLE", "BUS"}
DYN = ["vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"]
CLASS_KO = {"vehicle": "승용차 등 차량", "bus": "버스", "motorcyclist": "오토바이", "cyclist": "자전거",
            "pedestrian": "보행자", "none": "30m 안 없음"}
AHEAD_M = 40.0


def mark_kind(m):
    if m == "NONE":
        return "none"
    if "DASH_SOLID" in m or "SOLID_DASH" in m:
        return "mixed"
    if "DASH" in m:
        return "dashed"
    return "solid"


def load_lanes(sid):
    m = json.load(open(c.ROOT / sid / f"log_map_archive_{sid}.json"))
    lanes = {}
    for lid, l in m["lane_segments"].items():
        if l["lane_type"] not in LANE_TYPES:
            continue
        cl = np.array([[p["x"], p["y"]] for p in l["centerline"]])
        lanes[int(lid)] = dict(cl=cl, left=mark_kind(l["left_lane_mark_type"]), right=mark_kind(l["right_lane_mark_type"]),
                               succ=[int(s) for s in l["successors"]], inter=bool(l["is_intersection"]),
                               length=float(np.hypot(*np.diff(cl, axis=0).T).sum()))
    return lanes


def ego_lane(lanes, x, y, ux, uy):
    best, bd = None, np.inf
    for lid, l in lanes.items():
        cl = l["cl"]
        d = np.hypot(cl[:, 0] - x, cl[:, 1] - y)
        i = int(d.argmin())
        if d[i] > 3.0 or d[i] >= bd:
            continue
        j0, j1 = max(i - 1, 0), min(i + 1, len(cl) - 1)
        v = cl[j1] - cl[j0]
        n = np.hypot(*v)
        if n < 1e-6 or (v[0] * ux + v[1] * uy) / n < 0.7:
            continue
        best, bd = lid, d[i]
    return best


def lane_feats(lanes, lid):
    if lid is None:
        return dict(lane_found=0, left=np.nan, right=np.nan, lane_cat="차선 매칭 실패", dash_to_solid=np.nan, lane_inter=np.nan)
    l = lanes[lid]
    sides = {l["left"], l["right"]}
    if sides & {"dashed", "mixed"}:
        cat = "점선 경계 있음"
    elif "solid" in sides:
        cat = "실선만"
    else:
        cat = "표시 없음"
    # 앞 40m 안 후속 차선에서 같은 쪽이 실선으로 바뀌는가
    trans = 0
    for side in ["left", "right"]:
        if l[side] not in ("dashed", "mixed"):
            continue
        frontier, dist, seen = list(l["succ"]), l["length"] / 2, set()
        while frontier and dist <= AHEAD_M and not trans:
            nxt = []
            for s in frontier:
                if s in seen or s not in lanes:
                    continue
                seen.add(s)
                if lanes[s][side] == "solid":
                    trans = 1
                    break
                nxt += lanes[s]["succ"]
            dist += np.mean([lanes[s]["length"] for s in frontier if s in lanes] or [AHEAD_M])
            frontier = nxt
    return dict(lane_found=1, left=l["left"], right=l["right"], lane_cat=cat, dash_to_solid=trans, lane_inter=int(l["inter"]))


def nearest_class(g_t, x, y, r=30.0):
    o = g_t[(g_t["track_id"] != EGO) & g_t["object_type"].isin(DYN)]
    if o.empty:
        return "none", np.nan
    d = np.hypot(o["position_x"].to_numpy() - x, o["position_y"].to_numpy() - y)
    i = int(d.argmin())
    return (o["object_type"].iloc[i], float(d[i])) if d[i] <= r else ("none", np.nan)


def work(task):
    sid, items = task
    df = pd.read_parquet(c.ROOT / sid / f"scenario_{sid}.parquet", columns=c.COLS)
    lanes = load_lanes(sid)
    ego = df[df["track_id"] == EGO].set_index("timestep")
    rows = []
    for key, t in items:
        if t not in ego.index:
            continue
        e = ego.loc[t]
        x, y, vx, vy = float(e["position_x"]), float(e["position_y"]), float(e["velocity_x"]), float(e["velocity_y"])
        sp = np.hypot(vx, vy)
        lid = ego_lane(lanes, x, y, vx / sp, vy / sp) if sp >= 0.5 else None
        r = lane_feats(lanes, lid)
        r["near_class"], r["near_dist"] = nearest_class(df[df["timestep"] == t], x, y)
        r["key"] = key
        rows.append(r)
    return rows


def run(pairs):
    tasks = {}
    for sid, key, t in pairs:
        tasks.setdefault(sid, []).append((key, int(t)))
    out = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(work, list(tasks.items()), chunksize=20):
            out += r
    return pd.DataFrame(out)


def share_table(d, grp, col, levels, names):
    rows = []
    for lv in levels:
        a = d.loc[d[grp] == 1, col].eq(lv)
        b = d.loc[d[grp] == 0, col].eq(lv)
        o, lo, hi = E.woolf(int(a.sum()), len(a), int(b.sum()), len(b))
        rows.append({"구분": lv, names[0]: round(a.mean(), 3), names[1]: round(b.mean(), 3), "OR": o, "CI_low": lo, "CI_high": hi})
    return pd.DataFrame(rows).set_index("구분")


def main():
    tdir, fdir = Path("outputs/tables"), Path("outputs/figures")
    S = pd.read_parquet("outputs/ego_stops_feat.parquet").reset_index(drop=True)
    S["key"] = S.index
    S = S.merge(run(zip(S["scenario_id"], S["key"], S["t_onset"])), on="key", how="left")
    S["hard"] = (S["min_acc"] <= -3).astype(int)
    S.to_parquet("outputs/ego_stops_lane.parquet", index=False)

    cats = ["점선 경계 있음", "실선만", "표시 없음", "차선 매칭 실패"]
    print(f"[A1] 정지 에피소드 {len(S):,}건: 감속 시작 때 자차 차선 경계")
    t1 = share_table(S, "hard", "lane_cat", cats, ("급정지", "일반정지"))
    m = S[S["lane_found"] == 1]
    t1.loc["점선→실선 전환 구간(차선 매칭된 것 중)"] = list(share_table(m.assign(x=m["dash_to_solid"]), "hard", "x", [1], ("급정지", "일반정지")).iloc[0])
    print(t1.to_string())

    m = m.copy()
    m["is_event"] = m["hard"].astype(float)
    m["speed5"] = m["v_onset"] / 5
    m["near_cross8"] = (m["stop_dist_crosswalk"] <= 8).astype(float)
    m["dashed"] = (m["lane_cat"] == "점선 경계 있음").astype(float)
    m["unmarked"] = (m["lane_cat"] == "표시 없음").astype(float)
    m["d2s"] = m["dash_to_solid"].astype(float)
    X = np.column_stack([np.ones(len(m)), m["speed5"], m["near_cross8"], m["dashed"], m["unmarked"], m["d2s"]])
    b, se = L.logit(X, m["is_event"].to_numpy(float), m["scenario_id"].to_numpy())
    adj = pd.DataFrame({"OR": np.exp(b[1:]), "CI_low": np.exp(b[1:] - 1.96 * se[1:]), "CI_high": np.exp(b[1:] + 1.96 * se[1:])},
                       index=["감속 시작 속도 (+5 m/s)", "정지점 횡단보도 8m 이내", "점선 경계 있음 (vs 실선만)",
                              "표시 없음 (vs 실선만)", "점선→실선 전환 구간"]).round(2)
    print(f"\n[A2] 속도·횡단보도 보정 로지스틱 (차선 매칭된 {len(m):,}건)\n", adj.to_string())
    print(f"    '표시 없음' 차선 중 교차로 차선 비율: {m.loc[m['unmarked'] == 1, 'lane_inter'].mean():.1%}")
    pd.concat({"단순비교": t1, "보정": adj}).to_csv(tdir / "lane_marks_stops.csv")

    d = pd.read_parquet("outputs/ego_samples.parquet").reset_index(drop=True)
    d["key"] = d.index
    d = d.merge(run(zip(d["scenario_id"], d["key"], d["t"])), on="key", how="left")
    mm = d[d["lane_found"] == 1].assign(x=lambda z: z["dash_to_solid"])
    t2 = share_table(d, "is_event", "lane_cat", cats, ("이벤트", "대조"))
    t2.loc["점선→실선 전환 구간(차선 매칭된 것 중)"] = list(share_table(mm, "is_event", "x", [1], ("이벤트", "대조")).iloc[0])
    print(f"\n[A3] 자차 급감속 이벤트 {int(d['is_event'].sum())}건 vs 대조 {int((d['is_event'] == 0).sum())}개\n", t2.to_string())
    t2.to_csv(tdir / "lane_marks_events.csv")

    order = ["none", "vehicle", "bus", "pedestrian", "cyclist", "motorcyclist"]
    g = S.groupby("near_class")
    cj = pd.DataFrame({"건수": g.size(), "급정지 비율": g["hard"].mean(), "최소 저크 중앙값": g["min_jerk"].median(),
                       "최소 저크 하위 25%": g["min_jerk"].quantile(0.25), "최소 가속도 중앙값": g["min_acc"].median(),
                       "감속 시작 속도 중앙값": g["v_onset"].median()}).reindex(order).dropna(how="all").round(2)
    cj.index = [CLASS_KO[i] for i in cj.index]
    print("\n[B] 감속 시작 때 30m 안 최근접 객체 클래스별 자차 정지 저크\n", cj.to_string())
    cj.to_csv(tdir / "class_jerk.csv")

    keep = [k for k in order if (S["near_class"] == k).sum() >= 15]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, ref, unit in [(axes[0], "min_jerk", -2, "정지 중 최소 저크 (m/s³, -15 미만은 -15로 표시)"), (axes[1], "min_acc", -3, "정지 중 최소 가속도 (m/s²)")]:
        data = [S.loc[S["near_class"] == k, col].clip(lower=-15).to_numpy() for k in keep]
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=2.5, mfc="#999999", mec="none", alpha=0.5),
                        medianprops=dict(color="#222222", lw=1.6), whiskerprops=dict(color="#777777"),
                        capprops=dict(color="#777777"))
        for p in bp["boxes"]:
            p.set(facecolor="#cfe0fa", edgecolor="#2a6fdb")
        ax.axvline(ref, color="#222222", lw=1, ls="--")
        ax.set_yticks(range(1, len(keep) + 1), [f"{CLASS_KO[k]} (n={int((S['near_class'] == k).sum())})" for k in keep], fontsize=9)
        ax.set_xlabel(unit, fontsize=9, color="#777777")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e6e6")
        ax.set_axisbelow(True)
    axes[0].set_title("감속 시작 때 가장 가까운 객체(30m 안)별 자차 정지 저크", fontsize=11, loc="left")
    axes[1].set_title("같은 기준의 최소 가속도 (점선: 급정지 기준 -3)", fontsize=11, loc="left")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(fdir / "class_jerk_box.png", dpi=140, bbox_inches="tight")
    print("\n저장: outputs/figures/class_jerk_box.png")


if __name__ == "__main__":
    main()

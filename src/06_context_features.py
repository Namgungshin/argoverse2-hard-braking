"""D5: robust 급감속 이벤트와 같은 시나리오의 대조(non-event) 주행 시점에 맥락 변수를 붙인다.

변수:
- 지도: dist_lane(가장 가까운 차선 중심선까지, m), on_intersection(교차로 차선 위 여부),
        dist_intersection(교차로 차선 중심선까지 거리), dist_crosswalk(횡단보도까지 거리)
- 주변: n_agents_30m(30m 내 동적 객체 수), lead_gap(진행 방향 정면 선행 객체까지 중심 간 거리),
        lead_rel_speed(선행 속도 - 본인 속도), lead_ttc(접근 중일 때 충돌까지 시간)
- 품질: pre_max_acc(이벤트 시작 직전 1.5초 내 최대 가속도; 비현실적으로 크면 속도 펄스 의심)
대조군: 이벤트 후보(glitch 포함 전체)의 앞뒤 3초 밖, 속도 >= 3.2 m/s인 주행 시점을 이벤트당 2개 무작위 추출.

실행: python src/06_context_features.py [--limit N]
출력: outputs/context_samples.parquet
"""
import argparse
import importlib.util
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ex", "src/02_extract_events.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

ROOT = Path("data/val")
TARGET = ex.TARGET_TYPES
DYNAMIC = {"vehicle", "bus", "cyclist", "motorcyclist", "pedestrian"}
LANE_TYPES = {"VEHICLE", "BUS"}
MIN_SPEED_CTRL = 3.2     # robust 이벤트의 최소 시작 속도와 맞춤
EXCL = 30                # 대조군은 이벤트 후보 앞뒤 3초 밖
CTRL_PER_EVENT = 2
LEAD_MAX, LEAD_LAT, NEAR_R = 50.0, 2.0, 30.0
COLS = ["track_id", "object_type", "timestep", "position_x", "position_y",
        "velocity_x", "velocity_y", "scenario_id", "city"]


def load_map(sid):
    m = json.load(open(ROOT / sid / f"log_map_archive_{sid}.json"))
    pts, inter = [], []
    for ln in m["lane_segments"].values():
        if ln["lane_type"] not in LANE_TYPES:
            continue
        c = np.array([[p["x"], p["y"]] for p in ln["centerline"]])
        pts.append(c)
        inter.append(np.full(len(c), bool(ln["is_intersection"])))
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    inter = np.concatenate(inter) if inter else np.empty(0, dtype=bool)
    segs = []
    for pc in m["pedestrian_crossings"].values():
        e1, e2 = pc["edge1"], pc["edge2"]
        a = np.array([(e1[0]["x"] + e2[0]["x"]) / 2, (e1[0]["y"] + e2[0]["y"]) / 2])
        b = np.array([(e1[1]["x"] + e2[1]["x"]) / 2, (e1[1]["y"] + e2[1]["y"]) / 2])
        segs.append((a, b))
    return pts, inter, segs


def map_feats(x, y, mp):
    pts, inter, segs = mp
    out = dict(dist_lane=np.nan, on_intersection=np.nan, dist_intersection=np.nan, dist_crosswalk=np.nan)
    if len(pts):
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        i = int(d.argmin())
        out["dist_lane"] = float(d[i])
        out["on_intersection"] = float(bool(inter[i]) and d[i] <= 4.0)
        if inter.any():
            out["dist_intersection"] = float(d[inter].min())
    if segs:
        p = np.array([x, y])
        best = np.inf
        for a, b in segs:
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            best = min(best, float(np.hypot(*(a + t * ab - p))))
        out["dist_crosswalk"] = best
    return out


def surroundings(g_t, tid, x, y, vx, vy):
    sp = float(np.hypot(vx, vy))
    out = dict(n_agents_30m=0, lead_gap=np.nan, lead_rel_speed=np.nan, lead_ttc=np.nan)
    o = g_t[(g_t["track_id"] != tid) & g_t["object_type"].isin(DYNAMIC)]
    if o.empty:
        return out
    rx = o["position_x"].to_numpy() - x
    ry = o["position_y"].to_numpy() - y
    out["n_agents_30m"] = int((np.hypot(rx, ry) <= NEAR_R).sum())
    if sp < 1.0:
        return out
    ux, uy = vx / sp, vy / sp
    lon = rx * ux + ry * uy
    lat = np.abs(rx * uy - ry * ux)
    m = (lon > 0) & (lon <= LEAD_MAX) & (lat <= LEAD_LAT)
    if m.any():
        k = int(np.where(m)[0][lon[m].argmin()])
        ls = float(np.hypot(o["velocity_x"].to_numpy()[k], o["velocity_y"].to_numpy()[k]))
        out["lead_gap"] = float(lon[k])
        out["lead_rel_speed"] = ls - sp
        if ls - sp < -0.1:
            out["lead_ttc"] = float(lon[k] / (sp - ls))
    return out


def pre_accel(g, t):
    """해당 시점이 속한 연속 구간에서 (평활 속도, 직전 1.5초 내 최대 가속도, 구간 시작 후 경과 스텝)."""
    ts = g["timestep"].to_numpy()
    sp = np.hypot(g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy())
    cuts = np.where(np.diff(ts) != 1)[0] + 1
    for idx in np.split(np.arange(len(ts)), cuts):
        if ts[idx[0]] <= t <= ts[idx[-1]]:
            _, res = ex.segment_events(sp[idx])
            if res is None:
                return np.nan, np.nan, np.nan
            sm, acc, _ = res
            i = int(t - ts[idx[0]])
            return float(sm[i]), float(acc[max(i - 15, 0):i + 1].max()), i
    return np.nan, np.nan, np.nan


def make_row(sid, by_t, tracks, tid, t, is_event, mp):
    g_t = by_t.get(t)
    if g_t is None:
        return None
    r = g_t[g_t["track_id"] == tid]
    if r.empty:
        return None
    r = r.iloc[0]
    x, y, vx, vy = float(r["position_x"]), float(r["position_y"]), float(r["velocity_x"]), float(r["velocity_y"])
    sm, pre, pos = pre_accel(tracks[tid], t)
    row = dict(scenario_id=sid, track_id=tid, t=int(t), is_event=int(is_event),
               object_type=r["object_type"], city=r["city"],
               speed=float(np.hypot(vx, vy)), speed_smooth=sm, pre_max_acc=pre, seg_pos=pos)
    row.update(map_feats(x, y, mp))
    row.update(surroundings(g_t, tid, x, y, vx, vy))
    return row


def process_df(sid, df, events, windows, mp):
    by_t = {t: g for t, g in df.groupby("timestep")}
    tracks = {tid: g.sort_values("timestep") for tid, g in df.groupby("track_id")}
    rows = []
    for tid, t in events:
        r = make_row(sid, by_t, tracks, tid, t, True, mp)
        if r:
            rows.append(r)
    tg = df[df["object_type"].isin(TARGET)]
    tg = tg[np.hypot(tg["velocity_x"], tg["velocity_y"]) >= MIN_SPEED_CTRL]
    keep = np.ones(len(tg), dtype=bool)
    tids, ts = tg["track_id"].to_numpy(), tg["timestep"].to_numpy()
    for wtid, s, e in windows:
        keep &= ~((tids == wtid) & (ts >= s - EXCL) & (ts <= e + EXCL))
    cand = tg[keep]
    n = min(len(cand), CTRL_PER_EVENT * len(events))
    if n:
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        for i in rng.choice(len(cand), size=n, replace=False):
            c = cand.iloc[int(i)]
            r = make_row(sid, by_t, tracks, c["track_id"], int(c["timestep"]), False, mp)
            if r:
                rows.append(r)
    return rows


def work(task):
    sid, events, windows = task
    df = pd.read_parquet(ROOT / sid / f"scenario_{sid}.parquet", columns=COLS)
    return process_df(sid, df, events, windows, load_map(sid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rob = pd.read_parquet("outputs/events_robust.parquet")
    allev = pd.read_parquet("outputs/events.parquet", columns=["scenario_id", "track_id", "start_t", "end_t"])
    win = {sid: list(zip(g["track_id"], g["start_t"], g["end_t"])) for sid, g in allev.groupby("scenario_id")}
    ev_by = {sid: list(zip(g["track_id"], g["start_t"])) for sid, g in rob.groupby("scenario_id")}
    tasks = [(sid, e, win.get(sid, [])) for sid, e in ev_by.items()]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"시나리오 {len(tasks)}개 처리 시작 (이벤트 {sum(len(t[1]) for t in tasks):,}건)")

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=10), 1):
            rows += r
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)}")
    out = pd.DataFrame(rows)
    out.to_parquet("outputs/context_samples.parquet", index=False)

    print(f"\n샘플 {len(out):,}개 (이벤트 {int(out['is_event'].sum()):,} / 대조 {int((1 - out['is_event']).sum()):,})")
    print(f"좌표계 점검: 가장 가까운 차선까지 거리 중앙값 {out['dist_lane'].median():.2f} m (수 m 이내여야 정상)")
    g = out.groupby("is_event")
    print("\n이벤트(1) vs 대조(0) 요약")
    print(pd.DataFrame({
        "건수": g.size(),
        "속도 중앙값": g["speed_smooth"].median().round(1),
        "교차로 차선 위 비율": g["on_intersection"].mean().round(3),
        "횡단보도 거리 중앙값": g["dist_crosswalk"].median().round(1),
        "선행객체 있음 비율": g["lead_gap"].apply(lambda s: s.notna().mean()).round(3),
        "선행 거리 중앙값": g["lead_gap"].median().round(1),
        "주변객체수 중앙값": g["n_agents_30m"].median(),
        "pre_max_acc>4 비율": g["pre_max_acc"].apply(lambda s: (s > 4).mean()).round(3),
    }).T)


if __name__ == "__main__":
    main()

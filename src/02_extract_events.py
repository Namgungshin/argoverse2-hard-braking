"""Argoverse 2 Motion Forecasting 급감속 이벤트 추출 (v2).

v1 진단 결과 반영:
- 대상 객체를 움직이는 도로 이용자로 한정 (static/background/construction 제외)
- 중앙값 필터(5) -> 이동평균(3)으로 스파이크 노이즈 제거
- 원본 속도가 한 스텝에 15 m/s^2 이상 튀는 구간을 포함한 이벤트는 glitch=True 플래그

실행 예:
  python src/02_extract_events.py --limit 200
  python src/02_extract_events.py
결과: outputs/events.parquet, outputs/tracks.parquet
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.1                 # 10Hz
MED_WIN = 5              # 중앙값 필터 창 (0.5초)
SMOOTH_WIN = 3           # 이동평균 창 (0.3초)
THR = -3.0               # 급감속 임계값 (m/s^2)
THR_SEVERE = -4.5        # 심한 급감속
MIN_STEPS = 5            # 최소 지속 (0.5초)
MIN_SPEED = 2.0          # 이벤트 시작 속도 하한 (m/s)
GLITCH_STEP = 15.0       # 원본 속도 1스텝 변화율이 이 값(m/s^2)을 넘으면 트래킹 오류로 간주
GLITCH_PAD = 3           # 이벤트 앞뒤로 glitch를 검사할 여유 스텝
TARGET_TYPES = {"vehicle", "bus", "motorcyclist", "cyclist"}
COLS = ["track_id", "object_type", "object_category", "timestep", "observed",
        "velocity_x", "velocity_y", "scenario_id", "focal_track_id", "city"]


def find_runs(mask):
    """True가 연속된 구간의 (시작, 끝 포함) 인덱스 목록."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def segment_events(speed, thr=THR, min_steps=MIN_STEPS, min_speed=MIN_SPEED):
    """연속 구간 하나의 속도 배열에서 급감속 이벤트를 찾는다.

    반환: (events, (smoothed_speed, acc, raw_step_acc)) 이고
    events는 [(s, e, max_abs_raw_step_acc)] 목록.
    """
    n = len(speed)
    if n < max(min_steps, 3):
        return [], None
    s = pd.Series(speed)
    sm = (s.rolling(MED_WIN, center=True, min_periods=1).median()
           .rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy())
    acc = np.gradient(sm, DT)
    raw_step = np.diff(speed) / DT
    out = []
    for a, b in find_runs(acc <= thr):
        if b - a + 1 >= min_steps and sm[a] >= min_speed:
            lo, hi = max(a - GLITCH_PAD, 0), min(b + GLITCH_PAD, n - 1)
            out.append((a, b, float(np.abs(raw_step[lo:hi]).max())))
    return out, (sm, acc, raw_step)


def process_scenario(path):
    df = pd.read_parquet(path, columns=COLS)
    sid = df["scenario_id"].iloc[0]
    city = df["city"].iloc[0]
    focal = df["focal_track_id"].iloc[0]
    events, tracks = [], []
    for tid, g in df.sort_values("timestep").groupby("track_id", sort=False):
        otype = g["object_type"].iloc[0]
        ocat = int(g["object_category"].iloc[0])
        eligible = otype in TARGET_TYPES
        t = g["timestep"].to_numpy()
        speed = np.hypot(g["velocity_x"].to_numpy(), g["velocity_y"].to_numpy())
        obs = g["observed"].to_numpy()
        n_ev = n_clean = 0
        min_acc_track = np.nan
        n_glitch_steps = int((np.abs(np.diff(speed)) / DT > GLITCH_STEP).sum()) if len(speed) > 1 else 0
        if eligible:
            # 타임스텝이 끊긴 지점에서 구간 분리 (track fragment 등)
            cuts = np.where(np.diff(t) != 1)[0] + 1
            for idx in np.split(np.arange(len(t)), cuts):
                found, res = segment_events(speed[idx])
                if res is None:
                    continue
                sm, acc, _ = res
                min_acc_track = np.nanmin([min_acc_track, acc.min()])
                for a, b, max_step in found:
                    ac = acc[a:b + 1]
                    glitch = max_step > GLITCH_STEP
                    events.append(dict(
                        scenario_id=sid, city=city, track_id=tid,
                        object_type=otype, object_category=ocat,
                        is_focal=(tid == focal),
                        start_t=int(t[idx][a]), end_t=int(t[idx][b]),
                        duration_s=(b - a + 1) * DT,
                        min_acc=float(ac.min()), mean_acc=float(ac.mean()),
                        speed_start=float(sm[a]), speed_end=float(sm[b]),
                        speed_drop=float(sm[a] - sm[b]),
                        max_raw_step_acc=max_step,
                        glitch=bool(glitch),
                        started_in_observed=bool(obs[idx][a]),
                        severe=bool(ac.min() <= THR_SEVERE),
                    ))
                    n_ev += 1
                    n_clean += int(not glitch)
        tracks.append(dict(
            scenario_id=sid, city=city, track_id=tid,
            object_type=otype, object_category=ocat,
            is_focal=(tid == focal), eligible=eligible, n_steps=len(t),
            max_speed=float(speed.max()), min_acc=float(min_acc_track),
            n_glitch_steps=n_glitch_steps,
            n_events=n_ev, n_clean_events=n_clean,
        ))
    return events, tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/val")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted(Path(args.root).glob("*/scenario_*.parquet"))
    if args.limit:
        files = files[:args.limit]
    print(f"시나리오 {len(files)}개 처리 시작")

    all_ev, all_tr = [], []
    with Pool(args.workers) as pool:
        for i, (ev, tr) in enumerate(pool.imap_unordered(process_scenario, files, chunksize=20), 1):
            all_ev += ev
            all_tr += tr
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ev_df, tr_df = pd.DataFrame(all_ev), pd.DataFrame(all_tr)
    ev_df.to_parquet(out / "events.parquet", index=False)
    tr_df.to_parquet(out / "tracks.parquet", index=False)

    el = tr_df[tr_df["eligible"]]
    clean = ev_df[~ev_df["glitch"]]
    print(f"\n전체 객체 {len(tr_df):,}개 / 분석 대상 객체 {len(el):,}개")
    print(f"이벤트 {len(ev_df):,}건 중 glitch {ev_df['glitch'].mean():.1%} -> clean {len(clean):,}건")
    print(f"clean 이벤트가 있는 대상 객체 비율: {(el['n_clean_events'] > 0).mean():.2%}")
    print("\n유형별 clean 이벤트 보유 비율")
    print(el.groupby("object_type")["n_clean_events"].agg(
        objects="size", with_event=lambda s: (s > 0).sum()).assign(
        rate=lambda d: d["with_event"] / d["objects"]).sort_values("objects", ascending=False))
    if len(clean):
        print("\nclean 이벤트 요약")
        print(clean[["duration_s", "min_acc", "speed_start", "speed_drop"]].describe().round(2))
        bins = [-1000, -9.8, -6, -4.5, -3.0]
        labels = ["< -9.8", "-9.8~-6", "-6~-4.5", "-4.5~-3"]
        print("\nclean min_acc 구간 비율")
        print(pd.cut(clean["min_acc"], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels).round(3))


if __name__ == "__main__":
    main()

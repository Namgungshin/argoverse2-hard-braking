"""robust 이벤트 육안 검증: 상한(-9.8) 근처 이벤트 vs 중간 강도 이벤트의 원본 속도 시계열 비교.

실행: python src/05_inspect_samples.py
출력: outputs/figures/event_samples.png + 두 그룹의 수치 비교
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

spec = importlib.util.spec_from_file_location("ex", "src/02_extract_events.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

N = 8
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 이전과 다른 샘플을 보도록 기본 1
rob = pd.read_parquet("outputs/events_robust.parquet")
rob = rob[rob["object_type"] == "vehicle"]
groups = {
    "A: 상한 근처 (min_acc < -9)": rob[rob["min_acc"] < -9],
    "B: 중간 강도 (-6 ~ -4.5)": rob[(rob["min_acc"] >= -6) & (rob["min_acc"] < -4.5)],
}

print("그룹별 수치 비교")
for name, g in groups.items():
    print(f"  {name}: {len(g):,}건 | speed_start 중앙값 {g['speed_start'].median():.1f} | "
          f"speed_end<0.5 비율 {(g['speed_end'] < 0.5).mean():.1%} | "
          f"duration 중앙값 {g['duration_s'].median():.1f}s | 관측구간 시작 {g['started_in_observed'].mean():.1%}")

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for gi, (name, g) in enumerate(groups.items()):
    sample = g.sample(min(N, len(g)), random_state=SEED)
    for k, (_, r) in enumerate(sample.iterrows()):
        ax = axes[gi * 2 + k // 4, k % 4]
        p = Path(f"data/val/{r.scenario_id}/scenario_{r.scenario_id}.parquet")
        d = pd.read_parquet(p, columns=["track_id", "timestep", "velocity_x", "velocity_y"])
        d = d[d["track_id"] == r.track_id].sort_values("timestep")
        t = d["timestep"].to_numpy()
        sp = np.hypot(d["velocity_x"].to_numpy(), d["velocity_y"].to_numpy())
        _, res = ex.segment_events(sp)
        ax.plot(t, sp, ".-", color="gray", ms=3, lw=0.8, label="raw")
        if res is not None:
            ax.plot(t, res[0], color="tab:blue", lw=1.6, label="smoothed")
        ax.axvspan(r.start_t, r.end_t, color="tab:red", alpha=0.25)
        ax.set_title(f"{name[:1]} | acc {r.min_acc:.1f} | drop {r.speed_drop:.1f} m/s", fontsize=9)
        ax.set_xlabel("timestep")
        ax.set_ylabel("speed (m/s)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("행 1-2: A 그룹(상한 근처), 행 3-4: B 그룹(중간 강도) — 회색=원본, 파랑=평활, 빨강=이벤트 구간")
fig.tight_layout()
out = Path("outputs/figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "event_samples.png", dpi=130)
print("\n저장: outputs/figures/event_samples.png")

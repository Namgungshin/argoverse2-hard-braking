"""급감속 이벤트 품질 진단: 노이즈/물리적 불가능 값/경계 효과 확인."""
from pathlib import Path
import numpy as np
import pandas as pd

ev = pd.read_parquet("outputs/events.parquet")
tr = pd.read_parquet("outputs/tracks.parquet")
print(f"이벤트 {len(ev):,}건\n")

# 1. 가속도 크기 구간별 분포 (물리적 한계 약 -9.8 m/s^2)
bins = [-1000, -30, -15, -9.8, -6, -4.5, -3.0]
labels = ["< -30", "-30~-15", "-15~-9.8", "-9.8~-6", "-6~-4.5", "-4.5~-3"]
ev["acc_bin"] = pd.cut(ev["min_acc"], bins=bins, labels=labels)
print("[1] min_acc 구간별 비율 (전체 / 차량 / 정지류)")
static_types = ["static", "background", "construction", "riderless_bicycle"]
cmp = pd.DataFrame({
    "all": ev["acc_bin"].value_counts(normalize=True),
    "vehicle": ev[ev.object_type == "vehicle"]["acc_bin"].value_counts(normalize=True),
    "static류": ev[ev.object_type.isin(static_types)]["acc_bin"].value_counts(normalize=True),
}).reindex(labels[::-1]).round(3)
print(cmp, "\n")

# 2. 지속 시간 분포
print("[2] 지속 시간(초) 비율")
print(ev["duration_s"].round(1).value_counts(normalize=True).sort_index().head(8).round(3), "\n")

# 3. 경계 효과: 시나리오 처음/끝 근처에서 시작/종료하는 이벤트
edge = (ev["start_t"] <= 2) | (ev["end_t"] >= 107)
print(f"[3] 시나리오 경계(처음 2스텝/끝 3스텝) 이벤트 비율: {edge.mean():.1%}")
print(f"    경계 제외 시 -9.8 이하 비율: {(ev[~edge]['min_acc'] < -9.8).mean():.1%} (전체 {(ev['min_acc'] < -9.8).mean():.1%})\n")

# 4. 트랙 길이(짧은 fragment) 영향
m = ev.merge(tr[["scenario_id", "track_id", "n_steps"]], on=["scenario_id", "track_id"])
m["short"] = m["n_steps"] < 50
print("[4] 트랙 길이 50스텝 미만 비율 / 그 안에서 -9.8 이하 비율")
print(f"    {m['short'].mean():.1%} / {(m[m.short]['min_acc'] < -9.8).mean():.1%}  (긴 트랙: {(m[~m.short]['min_acc'] < -9.8).mean():.1%})\n")

# 5. 극단 이벤트 원본 속도 열 확인
print("[5] 극단 이벤트 5건의 원본 속도(m/s) 시계열")
worst = ev.nsmallest(5, "min_acc")
for _, r in worst.iterrows():
    p = Path(f"data/val/{r.scenario_id}/scenario_{r.scenario_id}.parquet")
    g = pd.read_parquet(p)
    g = g[g.track_id == r.track_id].sort_values("timestep")
    sp = np.hypot(g.velocity_x, g.velocity_y)
    sel = g.timestep.between(r.start_t - 3, r.end_t + 3)
    print(f"  {r.object_type} acc={r.min_acc:.1f} t={r.start_t}~{r.end_t}:",
          np.round(sp[sel].to_numpy(), 2).tolist())

# 6. 원본 속도의 한 스텝 변화량 노이즈 수준 (차량, 500 시나리오 표본)
files = sorted(Path("data/val").glob("*/scenario_*.parquet"))[:500]
jumps = []
for f in files:
    d = pd.read_parquet(f, columns=["track_id", "object_type", "timestep", "velocity_x", "velocity_y"])
    d = d[d.object_type == "vehicle"].sort_values(["track_id", "timestep"])
    d["sp"] = np.hypot(d.velocity_x, d.velocity_y)
    dd = d.groupby("track_id")["sp"].diff().dropna().abs() / 0.1
    jumps.append(dd.to_numpy())
j = np.concatenate(jumps)
print(f"\n[6] 차량 원본 속도 1스텝 변화율(m/s^2) 분위수: "
      f"50%={np.percentile(j,50):.2f} 90%={np.percentile(j,90):.2f} "
      f"99%={np.percentile(j,99):.2f}  >10m/s^2 비율={(j>10).mean():.2%}")

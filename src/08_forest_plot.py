"""context_logit.csv로 오즈비 포레스트 플롯 (모델 A 전체 vs 모델 B 정제).
도시 항목은 대조군을 같은 시나리오에서 뽑는 설계상 해석 불가라 제외한다.

실행: python src/08_forest_plot.py
출력: outputs/figures/context_forest.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("outputs/tables/context_logit.csv", index_col=0)
df = df[~df.index.str.startswith("도시")]
y = np.arange(len(df))[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
for off, suf, label, color in [(0.15, "A전체", "A: 전체 샘플", "#4c72b0"), (-0.15, "B정제", "B: 정제 샘플", "#dd8452")]:
    o, lo, hi = df[f"OR_{suf}"], df[f"CI_low_{suf}"], df[f"CI_high_{suf}"]
    ax.errorbar(o, y + off, xerr=[o - lo, hi - o], fmt="o", color=color, capsize=3, label=label)
ax.axvline(1, color="k", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 3])
ax.xaxis.set_major_formatter(mt.ScalarFormatter())
ax.set_yticks(y)
ax.set_yticklabels(df.index)
ax.set_xlabel("오즈비 (로그 축, 95% CI) — 1보다 크면 급감속 시작 가능성이 커짐")
ax.set_title("급감속 시작과 맥락 변수의 연관 (시나리오 클러스터 강건 SE)")
ax.legend()
fig.tight_layout()
out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/figures/context_forest.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("저장:", out)

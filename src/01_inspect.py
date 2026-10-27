"""시나리오 하나를 읽어 스키마와 속도/가속도 분포를 확인한다."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DT = 0.1  # 10Hz

def find_one(root="data/val"):
    return next(Path(root).glob("*/scenario_*.parquet"))

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_one()
    df = pd.read_parquet(path)
    print(path)
    print(df.dtypes, "\n")
    print(df.head(), "\n")
    print("객체 유형:", df.drop_duplicates("track_id")["object_type"].value_counts().to_dict())

    df["speed"] = np.hypot(df["velocity_x"], df["velocity_y"])
    for tid, g in df.sort_values("timestep").groupby("track_id"):
        acc = np.gradient(g["speed"].to_numpy(), DT)
        print(tid, g["object_type"].iloc[0], f"min acc={acc.min():.2f} m/s^2", f"steps={len(g)}")
        break

if __name__ == "__main__":
    main()

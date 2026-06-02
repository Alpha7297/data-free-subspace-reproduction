from pathlib import Path

import numpy as np
import pandas as pd


LENGTH = 100
ROOT = Path(__file__).resolve().parent


def read_series(name):
    path = ROOT / name
    data = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float64)
    if data.shape[1] != 2 * LENGTH:
        raise ValueError(f"{name}: expected {2 * LENGTH} columns, got {data.shape[1]}")
    return data.reshape(data.shape[0], LENGTH, 2)


def metric(data_free, explicit):
    n = min(len(data_free), len(explicit))
    data_free = data_free[:n]
    explicit = explicit[:n]
    diff = data_free - explicit
    point_l2 = np.linalg.norm(diff, axis=-1)
    frame_rms = np.sqrt(np.mean(point_l2 * point_l2, axis=-1))
    frame_max = np.max(point_l2, axis=-1)
    return {
        "frames": n,
        "mean_l2": float(np.mean(point_l2)),
        "first_frame_mean_l2": float(np.mean(point_l2[0])),
        "final_frame_mean_l2": float(np.mean(point_l2[-1])),
        "global_rms": float(np.sqrt(np.mean(point_l2 * point_l2))),
        "mean_frame_rms": float(np.mean(frame_rms)),
        "first_frame_rms": float(frame_rms[0]),
        "final_frame_rms": float(frame_rms[-1]),
        "max_frame_rms": float(np.max(frame_rms)),
        "max_point_error": float(np.max(frame_max)),
    }


def print_metric(name, m):
    print(name)
    print(f"  frames          : {m['frames']}")
    print(f"  mean L2         : {m['mean_l2']:.8g}")
    print(f"  first mean L2   : {m['first_frame_mean_l2']:.8g}")
    print(f"  final mean L2   : {m['final_frame_mean_l2']:.8g}")
    print(f"  global RMS      : {m['global_rms']:.8g}")
    print(f"  mean frame RMS  : {m['mean_frame_rms']:.8g}")
    print(f"  first frame RMS : {m['first_frame_rms']:.8g}")
    print(f"  final frame RMS : {m['final_frame_rms']:.8g}")
    print(f"  max frame RMS   : {m['max_frame_rms']:.8g}")
    print(f"  max point error : {m['max_point_error']:.8g}")


def check_conditions(data_free_pos, data_free_vel, explicit_pos, explicit_vel):
    print("condition check")
    print(f"  data_free pos shape: {data_free_pos.shape}")
    print(f"  explicit  pos shape: {explicit_pos.shape}")
    print(f"  data_free vel shape: {data_free_vel.shape}")
    print(f"  explicit  vel shape: {explicit_vel.shape}")
    print(f"  first saved pos RMS : {metric(data_free_pos[:1], explicit_pos[:1])['global_rms']:.8g}")
    print(f"  first saved vel RMS : {metric(data_free_vel[:1], explicit_vel[:1])['global_rms']:.8g}")
    print(f"  fixed point data_free pos[0,0]: {data_free_pos[0, 0]}")
    print(f"  fixed point explicit  pos[0,0]: {explicit_pos[0, 0]}")
    print(f"  fixed point data_free vel[0,0]: {data_free_vel[0, 0]}")
    print(f"  fixed point explicit  vel[0,0]: {explicit_vel[0, 0]}")


def main():
    data_free_pos = read_series("data_free_pos.csv")
    data_free_vel = read_series("data_free_vel.csv")
    explicit_pos = read_series("implicit_pos.csv")
    explicit_vel = read_series("implicit_vel.csv")

    check_conditions(data_free_pos, data_free_vel, explicit_pos, explicit_vel)
    print()
    print_metric("position error: data_free - explicit", metric(data_free_pos, explicit_pos))
    print()
    print_metric("velocity error: data_free - explicit", metric(data_free_vel, explicit_vel))


if __name__ == "__main__":
    main()

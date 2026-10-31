#!/usr/bin/env python3
"""离线比较 Raw 实测轨迹和已保存的回放采样，不连接设备。"""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="原始 Raw episode")
    parser.add_argument("--replay", type=Path, required=True, help="包含 replay_data.npz 的回放目录")
    args = parser.parse_args(argv)

    import numpy as np

    from dexmani_real.replay.metrics import compute_metrics
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.utils.io import write_json

    trajectory = load_trajectory(args.episode)
    with np.load(args.replay / "replay_data.npz", allow_pickle=False) as saved:
        measurements = {name: saved[name] for name in ("arm_qpos", "hand_qpos")}
    metrics = compute_metrics(trajectory, measurements)
    output = args.replay / "metrics.json"
    write_json(metrics, output)
    print(f"Replay: {metrics['matching_frames']}/{metrics['original_frames']} matching frames")
    for name in ("arm_joint", "hand_joint", "eef_position", "eef_orientation"):
        print(f"  {name}: {metrics[name]}")
    print(f"Metrics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

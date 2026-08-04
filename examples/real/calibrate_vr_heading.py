#!/usr/bin/env python3
"""One-shot VR heading calibration: compute the fixed T_vr_to_robot matrix.

Starts a VR receiver, collects wrist (or head) orientation data, computes
the mean forward direction, and writes ``vr_transform.json`` to
``dexmani_real/config/``.

Usage::

    python examples/real/calibrate_vr_heading.py [--duration 10] [--ref wrist]

By default, uses **wrist** orientation — the operator extends their arm and
points their fingers toward robot +X.  The wrist forward direction directly
reflects the hand motion that controls the robot, and is unaffected by head
tilt.  Pass ``--ref head`` to use head orientation instead.

Accuracy improvements over v1:
- Wrist-based (default): directly measures the hand's forward direction
- Countdown (3-2-1): allows operator to settle into position
- Outlier rejection: >3σ from circular mean are discarded
- Quality grading: excellent (σ<2°) / good (σ<5°) / poor (σ>5°)
- Longer default duration: 10s for better averaging
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dexmani_real.planning.pose_utils import normalize_quat_wxyz
from dexmani_real.sensor.vr_receiver_process import vr_loop
from dexmani_real.shm.shared_storage import SharedStorage

CONFIG_DIR = _repo_root / "dexmani_real" / "config"
OUTPUT_PATH = CONFIG_DIR / "vr_transform.json"
AUDIO_PATH = _repo_root / "assets" / "audio" / "轴向已标定.wav"


# ── helpers ──────────────────────────────────────────────────────────────────


def _forward_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """FLU +X (forward) direction from quaternion (wxyz)."""
    r = Rotation.from_quat(np.roll(q, -1))  # wxyz → xyzw
    return r.apply(np.array([1.0, 0.0, 0.0]))


def _circular_mean(forwards: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Circular mean of 2D forward directions with outlier rejection.

    Returns:
        theta_rad: mean heading angle in radians.
        mean_fwd: mean unit 2D vector.
        inlier_mask: boolean mask of frames kept.
    """
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    mask = norms >= 1e-6
    if not np.all(mask):
        n_bad = int(np.sum(~mask))
        print(f"  WARNING: {n_bad} 帧 forward 近乎垂直 (已跳过)")
    fwd_2d = fwd_2d[mask]
    norms = norms[mask]
    fwd_unit = fwd_2d / norms[:, None]

    # First pass: unweighted circular mean
    mean_0 = np.mean(fwd_unit, axis=0)
    mean_0 /= np.linalg.norm(mean_0)

    # Outlier rejection: |sin(Δθ)| ≈ angular distance from mean
    dists = np.abs(np.cross(fwd_unit, mean_0))
    thresh = 3.0 * np.std(dists)
    inlier = dists <= thresh
    n_out = int(np.sum(~inlier))
    if n_out > 0:
        print(f"  INFO: 剔除 {n_out} 个离群帧 (>3σ)")

    # Second pass: mean of inliers only
    fwd_inlier = fwd_unit[inlier]
    mean_fwd = np.mean(fwd_inlier, axis=0)
    mean_fwd /= np.linalg.norm(mean_fwd)
    theta = float(np.arctan2(mean_fwd[1], mean_fwd[0]))

    # Remap M-length inlier back to N-length array so callers can index
    # against the original forwards array without shape mismatch.
    full_inlier = np.zeros(len(forwards), dtype=bool)
    full_inlier[mask] = inlier
    return theta, mean_fwd, full_inlier


def _quality_grade(forwards: np.ndarray, theta_mean: float, inlier: np.ndarray) -> str:
    """Grade calibration quality from per-frame theta scatter."""
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    mask = (norms >= 1e-6) & inlier
    fwd_unit = fwd_2d[mask] / norms[mask, None]
    thetas = np.arctan2(fwd_unit[:, 1], fwd_unit[:, 0])
    dtheta = np.angle(np.exp(1j * (thetas - theta_mean)))
    std_deg = float(np.rad2deg(np.std(dtheta)))
    max_dev = float(np.rad2deg(np.max(np.abs(dtheta))))

    if std_deg < 2.0:
        grade = "excellent"
    elif std_deg < 5.0:
        grade = "good"
    else:
        grade = "POOR"
    return f"{grade} (σ={std_deg:.1f}°, max={max_dev:.1f}°)"


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="VR heading calibration")
    parser.add_argument("--duration", type=float, default=10.0, help="采集时长 (秒), default: 10")
    parser.add_argument("--port", type=int, default=8000, help="VR TCP 端口, default: 8000")
    parser.add_argument(
        "--ref",
        choices=["wrist", "head"],
        default="wrist",
        help="标定参考: wrist=手腕指向机器人+X (默认), head=头部面朝机器人+X",
    )
    args = parser.parse_args()

    ref_label = "手腕 (伸出右臂, 手指指向机器人 +X)" if args.ref == "wrist" else "头部 (面朝机器人 +X)"

    print("=" * 55)
    print("VR Heading 标定")
    print(f"  参考:   {ref_label}")
    print(f"  采集:   {args.duration}s")
    print(f"  端口:   {args.port}")
    print("=" * 55)

    # ── Start VR receiver (new architecture: SharedStorage + vr_loop) ──
    import multiprocessing as mp

    shared = SharedStorage.create(prefix="dexmani_vr_calib")
    from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig
    vr_cfg = VRReceiverConfig(port=args.port)
    vr_proc = mp.Process(target=vr_loop, args=(shared, vr_cfg), name="vr-calib", daemon=True)
    vr_proc.start()

    if not shared.vr_ready.wait(timeout=15):
        print("ERROR: VR receiver 启动失败 (ready timeout)")
        shared.is_running.value = False
        vr_proc.join(timeout=5)
        shared.close()
        sys.exit(1)

    print("\nVR receiver 已启动, 等待数据...")

    # Wait for first valid frame (up to 120s)
    deadline = time.monotonic() + 120.0
    last_print = 0.0
    while time.monotonic() < deadline:
        result = shared.vr_ring.read_latest()
        if result is not None:
            data, _ts, _seq = result
            hp = np.asarray(data["head_pos"][0], dtype=np.float64)
            if np.any(hp != 0):
                print("  已收到数据\n")
                break
        now = time.monotonic()
        if now - last_print >= 5.0:
            print(f"  等待中... ({int(now - (deadline - 120.0))}s, 请确认Quest app正在运行)")
            last_print = now
        time.sleep(0.1)
    else:
        print("ERROR: 120s内未收到数据, 请确认VR已连接且Quest app正在运行")
        shared.is_running.value = False
        vr_proc.join(timeout=5)
        shared.close()
        sys.exit(1)

    # ── Countdown ──
    if args.ref == "wrist":
        print("  请伸出右臂, 手指指向机器人 +X 方向, 保持稳定...")
    else:
        print("  请面朝机器人 +X 方向, 保持头部静止...")
    for i in [3, 2, 1]:
        print(f"  {i}...")
        time.sleep(1.0)

    # ── Collect ──
    forwards: list[np.ndarray] = []
    deadline = time.monotonic() + args.duration
    last_print = 0.0
    quat_key = "wrist_quat_wxyz" if args.ref == "wrist" else "head_quat_wxyz"

    print(f"  采集 {args.duration}s (保持静止)...")
    while time.monotonic() < deadline:
        result = shared.vr_ring.read_latest()
        if result is None:
            time.sleep(0.01)
            continue

        data, _ts, _seq = result

        # Extract quaternion from structured array field.
        if args.ref == "wrist":
            q = np.asarray(data["wrist_quat_wxyz"][0], dtype=np.float64)
        else:
            q = np.asarray(data["head_quat_wxyz"][0], dtype=np.float64)

        if not np.all(np.isfinite(q)):
            continue

        # For head mode: skip frames without valid head position
        if args.ref == "head":
            hp = np.asarray(data["head_pos"][0], dtype=np.float64)
            if not np.any(hp != 0):
                continue

        q = normalize_quat_wxyz(q)
        forwards.append(_forward_from_quat_wxyz(q))

        now = time.monotonic()
        if now - last_print >= 1.0:
            print(f"    已采集 {len(forwards)} 帧...")
            last_print = now
        time.sleep(0.01)

    shared.is_running.value = False
    vr_proc.join(timeout=5)
    if vr_proc.is_alive():
        vr_proc.terminate()
        vr_proc.join(timeout=1)
    shared.close()

    if len(forwards) < 30:
        print(f"ERROR: 只采集到 {len(forwards)} 帧 ({args.ref} quat 不可用?), 样本不足 (需 ≥30)")
        sys.exit(1)

    # ── Compute ──
    forwards_arr = np.array(forwards, dtype=np.float64)
    theta_rad, mean_fwd, inlier = _circular_mean(forwards_arr)
    theta_deg = float(np.rad2deg(theta_rad))
    quality = _quality_grade(forwards_arr, theta_rad, inlier)

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    T = np.array(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Sanity check
    corrected = T @ np.array([mean_fwd[0], mean_fwd[1], 0.0])

    print(f"\n{'='*55}")
    print(f"标定结果")
    print(f"  有效帧:   {len(forwards)} (剔除 {int(np.sum(~inlier))} 离群)")
    print(f"  forward:  [{mean_fwd[0]:.4f}, {mean_fwd[1]:.4f}]")
    print(f"  theta:    {theta_deg:.1f}°")
    print(f"  质量:     {quality}")
    print(f"  验证:     T·forward = [{corrected[0]:.4f}, {corrected[1]:.4f}] (want [1, 0])")
    print(f"  T = R_z(-{theta_deg:.1f}°):")
    print(f"    [{T[0,0]:.4f}, {T[0,1]:.4f}, {T[0,2]:.4f}],")
    print(f"    [{T[1,0]:.4f}, {T[1,1]:.4f}, {T[1,2]:.4f}],")
    print(f"    [{T[2,0]:.4f}, {T[2,1]:.4f}, {T[2,2]:.4f}]")
    if corrected[0] < 0.98:
        print(f"  WARNING: 校正偏差大, 请重新标定!")
    print(f"{'='*55}")

    # ── Write config ──
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "description": "Fixed VR-to-robot transform (LeFranX/ManiUniCon style)",
        "T_vr_to_robot": T.tolist(),
        "theta_deg": theta_deg,
        "convention": "R_z(-theta) maps VR FLU forward -> robot base +X",
        "ref": args.ref,
        "quality": quality,
        "frames": len(forwards),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n已保存到: {OUTPUT_PATH}")

    # ── Audio ──
    if AUDIO_PATH.exists():
        player = "aplay" if sys.platform == "linux" else "afplay"
        subprocess.run([player, str(AUDIO_PATH)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  音频反馈已播放")


if __name__ == "__main__":
    main()

"""Episode health check — offline data-quality report for recorded HDF5 episodes.

Scans one or more episodes and reports the metrics that expose silent
collection failures:

- Grid fill: consecutive-duplicate /timestamp values = forward-filled slots
  (decision-loop overruns; meta fps only reflects the grid rate).
- Camera content duplication: per-frame ROI hash over /rgb — cam-writer
  backlog drops repeat frames on disk while flag_camera_fresh stays True
  (it only tracks SHM arrival, not what reached the file).
- Tracking error: max|/action_arm_joint - /arm_qpos| per frame — mode-6
  speed-saturation lag between commanded and actual joints.
- Flags summary + /meta echo (v8 attrs: truncated, stop_reason,
  cam_frames_dropped, cam_items_written; schema <= 7 files print "-").

Usage:
    python -m dexmani_real.tools.check_episode_health episodes/episode_*.h5
    python -m dexmani_real.tools.check_episode_health ep.h5 --roi-stride 8 --track-thresh-rad 0.35

Read-only. Camera frames are read one at a time (never loaded whole) so
multi-GB episodes are safe; a full rgb scan of a 60 s episode takes a while.
Exit code 1 when any WARN fired (usable as a collection gate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

ASSUMED_CAMERA_FPS = 30.0  # RealSense capture rate — drives the expected-dup baseline

# WARN thresholds
FILL_WARN_PCT = 10.0  # forward-filled grid slots
CAM_DUP_WARN_MARGIN_PCT = 15.0  # camera dup% above the rate-derived baseline
TRACK_P95_WARN_DEG = 20.0
FREEZE_REPORT_MIN_S = 0.2  # only list camera freeze runs at least this long


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, length) of each run of consecutive True in ``mask``."""
    runs: list[tuple[int, int]] = []
    n = 0
    for i, v in enumerate(mask):
        if v:
            n += 1
        elif n:
            runs.append((i - n, n))
            n = 0
    if n:
        runs.append((len(mask) - n, n))
    return runs


def _check_grid(f: h5py.File, control_hz: float) -> float:
    """Print grid fill stats; return fill percentage."""
    if "timestamp" not in f:
        print("  栅格: /timestamp 缺失 — 跳过")
        return 0.0
    ts = f["timestamp"][:]
    if len(ts) < 2:
        print(f"  栅格: 仅 {len(ts)} 帧 — 跳过")
        return 0.0
    dup = np.diff(ts) == 0.0  # forward-filled slots repeat the previous raw value
    fill_pct = 100.0 * dup.sum() / len(ts)
    dt = 1.0 / control_hz if control_hz > 0 else float("nan")
    longest = max((n for _, n in _runs_of(dup)), default=0)
    print(
        f"  栅格填充: {int(dup.sum())}/{len(ts)} 槽 ({fill_pct:.1f}%)  "
        f"最长连跑 {longest} 槽 ≈ {longest * dt * 1000:.0f}ms"
    )
    return fill_pct


def _check_camera(f: h5py.File, key: str, stride: int, control_hz: float) -> tuple[float, float]:
    """Print content-duplication stats for one camera dataset.

    Returns (dup_pct, expected_pct). Frames are hashed one at a time on a
    ``::stride`` ROI — never ``[:]`` (episodes are multi-GB).
    """
    ds = f[key]
    T = ds.shape[0]
    dup = np.zeros(T, dtype=bool)
    prev: int | None = None
    for t in range(T):
        h = hash(ds[t, ::stride, ::stride].tobytes())
        dup[t] = prev is not None and h == prev
        prev = h
        if T > 1000 and t > 0 and t % 1000 == 0:
            print(f"    …{key} 扫描 {t}/{T}", flush=True)
    dup_pct = 100.0 * dup[1:].sum() / max(1, T - 1)
    # Baseline: a 30fps camera on a control_hz grid legitimately repeats
    # max(0, 1 - 30/control_hz) of rows (0% @16Hz, 40% @50Hz).
    expected_pct = 100.0 * max(0.0, 1.0 - ASSUMED_CAMERA_FPS / control_hz) if control_hz > 0 else 0.0
    dt = 1.0 / control_hz if control_hz > 0 else float("nan")
    freezes = sorted(_runs_of(dup), key=lambda r: -r[1])
    freeze_desc = "  ".join(f"{n * dt:.2f}s@t={s * dt:.1f}s" for s, n in freezes[:3] if n * dt >= FREEZE_REPORT_MIN_S)
    n_freezes = sum(1 for _, n in freezes if n * dt >= FREEZE_REPORT_MIN_S)
    print(
        f"  {key} 内容重复: {dup_pct:.1f}% (期望基线 {expected_pct:.0f}% @{control_hz:.0f}Hz/30fps)  "
        f"冻结段≥{FREEZE_REPORT_MIN_S:.1f}s: {n_freezes}" + (f"  最长: {freeze_desc}" if freeze_desc else "")
    )
    return dup_pct, expected_pct


def _check_tracking(f: h5py.File, thresh_rad: float) -> float | None:
    """Print cmd-state tracking-error stats; return p95 in degrees."""
    if "action_arm_joint" not in f or "arm_qpos" not in f:
        print("  跟踪误差: action_arm_joint/arm_qpos 缺失 — 跳过")
        return None
    err = np.max(np.abs(f["action_arm_joint"][:] - f["arm_qpos"][:]), axis=1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        print("  跟踪误差: 全 NaN — 跳过")
        return None
    p95 = float(np.percentile(err, 95))
    over = 100.0 * float(np.mean(err > thresh_rad))
    print(
        f"  跟踪误差 |cmd-state|∞: mean {np.degrees(err.mean()):.1f}°  "
        f"p95 {np.degrees(p95):.1f}°  max {np.degrees(err.max()):.1f}°  "
        f">{np.degrees(thresh_rad):.0f}°: {over:.1f}% 帧"
    )
    return float(np.degrees(p95))


def check_episode(path: str, roi_stride: int, track_thresh_rad: float) -> list[str]:
    """Run all checks on one episode; return the WARN lines."""
    warns: list[str] = []
    with h5py.File(path, "r") as f:
        meta = dict(f["meta"].attrs) if "meta" in f else {}
        control_hz = float(meta.get("control_hz", 0.0) or 0.0)

        print(f"\n=== {path} ===")
        print(
            f"  meta: schema={meta.get('schema_version', '-')}  control_hz={control_hz:g}  "
            f"frames={meta.get('num_frames', '-')}  dur={meta.get('duration', float('nan')):.1f}s  "
            f"success={meta.get('success', '-')}  min_frames_met={meta.get('min_frames_met', '-')}"
        )
        print(
            f"        truncated={meta.get('truncated', '-')}  stop_reason={meta.get('stop_reason', '-')}  "
            f"cam_dropped={meta.get('cam_frames_dropped', '-')}  cam_written={meta.get('cam_items_written', '-')}"
        )

        fill_pct = _check_grid(f, control_hz)
        if fill_pct > FILL_WARN_PCT:
            warns.append(f"栅格填充 {fill_pct:.1f}% > {FILL_WARN_PCT:.0f}% — 决策循环跟不上 control_hz")

        cam_keys = [k for k in f.keys() if k == "rgb" or k.endswith("_rgb")]
        for key in cam_keys:
            dup_pct, expected_pct = _check_camera(f, key, roi_stride, control_hz)
            if dup_pct > expected_pct + CAM_DUP_WARN_MARGIN_PCT:
                warns.append(f"{key} 内容重复 {dup_pct:.1f}% 超基线 {expected_pct:.0f}% — cam-writer 丢帧/相机停流")
        if not cam_keys:
            print("  相机: 无 rgb 数据集 — 跳过")

        dropped = int(meta.get("cam_frames_dropped", 0) or 0)
        if dropped > 0:
            warns.append(f"cam_frames_dropped={dropped} — 写线程队列丢帧 (内容被前向填充)")

        p95_deg = _check_tracking(f, track_thresh_rad)
        if p95_deg is not None and p95_deg > TRACK_P95_WARN_DEG:
            warns.append(f"跟踪误差 p95 {p95_deg:.1f}° > {TRACK_P95_WARN_DEG:.0f}° — 臂速度饱和，动作标签失真")

        flags = [k for k in ("flag_ik_ok", "flag_held", "flag_camera_fresh", "flag_retarget_ok") if k in f]
        if flags:
            print("  flags: " + "  ".join(f"{k}={100.0 * float(np.mean(f[k][:])):.1f}%" for k in flags))

        if bool(meta.get("truncated", False)):
            warns.append(f"truncated=True (stop_reason={meta.get('stop_reason', '-')}) — 撞 max_frames 截断")

    for w in warns:
        print(f"  ⚠ WARN: {w}")
    if not warns:
        print("  ✓ 无警告")
    return warns


def main() -> None:
    parser = argparse.ArgumentParser(description="DexMani episode health check (read-only)")
    parser.add_argument("episodes", nargs="+", help="HDF5 episode file(s)")
    parser.add_argument("--roi-stride", type=int, default=8, help="Pixel stride for the rgb content hash (default: 8).")
    parser.add_argument(
        "--track-thresh-rad",
        type=float,
        default=0.35,
        help="Tracking-error threshold in rad for the over-threshold ratio (default: 0.35 = inner-loop warn).",
    )
    args = parser.parse_args()

    total_warns = 0
    for ep in args.episodes:
        path = Path(ep).expanduser().resolve()
        if not path.is_file():
            logger.error("File not found: %s", path)
            total_warns += 1
            continue
        try:
            total_warns += len(check_episode(str(path), args.roi_stride, args.track_thresh_rad))
        except (OSError, KeyError, ValueError) as e:
            logger.error("Failed to check %s: %s", path, e)
            total_warns += 1

    if len(args.episodes) > 1:
        print(f"\n共 {len(args.episodes)} 个文件, {total_warns} 项警告")
    sys.exit(1 if total_warns else 0)


if __name__ == "__main__":
    main()

"""Episode health check — offline data-quality report for recorded episodes.

Scans one or more episodes and reports the metrics that expose silent
collection failures:

- Grid fill: consecutive-duplicate /timestamp values = forward-filled slots
  (decision-loop overruns; meta fps only reflects the grid rate).
- Camera content duplication: per-frame ROI hash over /rgb — detects
  repeated pixel data from cam-writer backlog drops.
- Arm tracking error: max|/action_arm_joint - /arm_qpos| per frame — mode-6
  speed-saturation lag between commanded and actual joints.
- Hand tracking error: max|/action_hand_joint - /hand_qpos| per frame —
  commanded vs actual hand joint positions.
- Flags summary + /meta echo (v8 attrs: truncated, stop_reason,
  cam_frames_dropped, cam_items_written; schema <= 7 files print "-").

Supports both legacy (single ``.h5``) and new (directory with ``data.h5`` +
``depth.h5`` + ``rgb.mp4``) episode formats.

Usage:
    python -m dexmani_real.tools.check_episode_health episodes/episode_*.h5
    python -m dexmani_real.tools.check_episode_health episodes/episode_dir/
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

from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

ASSUMED_CAMERA_FPS = 30.0  # RealSense capture rate — drives the expected-dup baseline

# WARN thresholds
FILL_WARN_PCT = 10.0  # forward-filled grid slots
CAM_DUP_WARN_MARGIN_PCT = 15.0  # camera dup% above the rate-derived baseline
TRACK_P95_WARN_DEG = 20.0
FREEZE_REPORT_MIN_S = 0.2  # only list camera freeze runs at least this long
HAND_TRACK_P95_WARN_DEG = 20.0
TACTILE_ALLZERO_WARN_PCT = 90.0  # warn if >X% of frames have zero force on a finger
TACTILE_TIPBOARD_ERR_WARN = 1  # warn if any tipboard_err non-zero


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


def _check_camera(data: np.ndarray | h5py.Dataset, key: str, stride: int, control_hz: float) -> tuple[float, float]:
    """Print content-duplication stats for one camera dataset.

    Returns (dup_pct, expected_pct).  ``data`` may be a pre-decoded numpy
    array (video sidecar path) or an ``h5py.Dataset`` (legacy HDF5 path) —
    both support ``[t, ::stride, ::stride]`` indexing and ``.tobytes()``.

    Frames are hashed one at a time on a ``::stride`` ROI — never ``[:]``
    (episodes are multi-GB on the legacy path).
    """
    T = data.shape[0]
    dup = np.zeros(T, dtype=bool)
    prev: int | None = None
    for t in range(T):
        h = hash(data[t, ::stride, ::stride].tobytes())
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


def _check_hand_tracking(f: h5py.File, thresh_rad: float) -> float | None:
    """Print hand cmd-state tracking-error stats; return p95 in degrees."""
    if "action_hand_joint" not in f or "hand_qpos" not in f:
        print("  手跟踪误差: action_hand_joint/hand_qpos 缺失 — 跳过")
        return None
    err = np.max(np.abs(f["action_hand_joint"][:] - f["hand_qpos"][:]), axis=1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        print("  手跟踪误差: 全 NaN — 跳过")
        return None
    p95 = float(np.percentile(err, 95))
    over = 100.0 * float(np.mean(err > thresh_rad))
    print(
        f"  手跟踪误差 |cmd-state|∞: mean {np.degrees(err.mean()):.1f}°  "
        f"p95 {np.degrees(p95):.1f}°  max {np.degrees(err.max()):.1f}°  "
        f">{np.degrees(thresh_rad):.0f}°: {over:.1f}% 帧"
    )
    return float(np.degrees(p95))


def _check_hand_freeze(f: h5py.File, control_hz: float) -> list[tuple[int, int, float]]:
    """Detect hand qpos freeze runs (driver board lockout).

    A freeze run is a span where hand_qpos is constant (max joint delta
    < 1e-4 rad) for >= *min_frames* while action_hand_joint is actively
    changing (max joint delta >= 0.05 rad), indicating that commands are
    being sent but the hand is not executing them.

    Returns list of (start_frame, length, max_cmd_gap_deg).
    """
    if "hand_qpos" not in f or "action_hand_joint" not in f:
        return []

    hq = f["hand_qpos"][:]  # (T, 12)
    cmd = f["action_hand_joint"][:]  # (T, 12)
    min_frames = max(8, int(control_hz * 0.5))  # 0.5 s
    cmd_active_thresh_rad = 0.05  # ~2.9°

    hq_step = np.max(np.abs(np.diff(hq, axis=0)), axis=1)  # (T-1,)
    cmd_step = np.max(np.abs(np.diff(cmd, axis=0)), axis=1)  # (T-1,)

    # A frame is "frozen" if qpos didn't change from the previous frame
    frozen_mask = np.concatenate([[False], hq_step < 1e-4])  # (T,)

    freeze_runs: list[tuple[int, int, float]] = []
    for start, length in _runs_of(frozen_mask):
        if length < min_frames:
            continue
        end = start + length
        # Check if cmd was active during this freeze
        cmd_slice = cmd_step[max(0, start - 1):min(len(cmd_step), end)]
        cmd_active = bool(np.any(cmd_slice >= cmd_active_thresh_rad))
        if not cmd_active:
            continue
        # Compute max cmd-qpos gap during the freeze
        gap = np.max(np.abs(cmd[start:end] - hq[start:end]))
        freeze_runs.append((start, length, float(np.rad2deg(gap))))

    return freeze_runs


def _check_tactile(f: h5py.File) -> dict[str, float | None]:
    """Print tactile data health stats; return per-finger zero-force percentages.

    Checks:
      - Whether tactile datasets exist (hand_tactile_force, hand_contact)
      - Per-finger force all-zero ratio (inactive sensor warning)
      - NaN contamination
      - tipboard_err non-zero presence (hardware sensor fault flag)
    """
    SENSOR_NAMES = ["thumb", "index", "middle", "ring", "little"]

    result: dict[str, float | None] = {name: None for name in SENSOR_NAMES}

    # ── Per-finger force from hand_contact (5,3) ──
    if "hand_contact" not in f:
        print("  触觉: hand_contact 缺失 — 跳过")
        return result

    force_sum = f["hand_contact"][:]  # (T,5,3)
    if force_sum.size == 0:
        print("  触觉: hand_contact 为空 — 跳过")
        return result

    T = force_sum.shape[0]
    has_nan = not np.all(np.isfinite(force_sum))
    nan_pct = 100.0 * float(np.mean(~np.isfinite(force_sum))) if has_nan else 0.0

    # Per-finger L2 norm → zero ratio
    force_mag = np.linalg.norm(force_sum, axis=2)  # (T,5)
    zero_pcts = {}
    for i, name in enumerate(SENSOR_NAMES):
        zero_pct = 100.0 * float(np.mean(force_mag[:, i] == 0.0))
        zero_pcts[name] = zero_pct
        result[name] = zero_pct

    zero_desc = "  ".join(f"{name}={zero_pcts[name]:.1f}%" for name in SENSOR_NAMES)
    print(f"  触觉力零值率: {zero_desc}")
    if has_nan:
        print(f"  触觉 NaN 比例: {nan_pct:.2f}%")

    # ── Per-taxel force from hand_tactile_force (5,120,3) ──
    if "hand_tactile_force" in f:
        tactile_force = f["hand_tactile_force"][:]  # (T,5,120,3)
        if tactile_force.size > 0:
            taxel_mag = np.linalg.norm(tactile_force, axis=3)  # (T,5,120)
            # Aggregate across time: mean per taxel
            mean_taxel_mag = np.mean(taxel_mag, axis=0)  # (5,120)
            active_taxels_per_finger = np.sum(mean_taxel_mag > 0.0, axis=1)  # (5,)
            for i, name in enumerate(SENSOR_NAMES):
                print(f"    {name}: {active_taxels_per_finger[i]}/120 active taxels  "
                      f"max_taxel_mag={np.max(mean_taxel_mag[i]):.3f} N")
    else:
        print("  触觉: hand_tactile_force 缺失 — 跳过细粒度 taxel 检查")

    # ── Contact boolean (hand_tactile_contact) ──
    if "hand_tactile_contact" in f:
        contact = f["hand_tactile_contact"][:]  # (T,5) bool
        contact_pcts = {}
        for i, name in enumerate(SENSOR_NAMES):
            contact_pcts[name] = 100.0 * float(np.mean(contact[:, i])) if contact.shape[0] > 0 else 0.0
        contact_desc = "  ".join(f"{name}={contact_pcts[name]:.1f}%" for name in SENSOR_NAMES)
        print(f"  触觉接触率: {contact_desc}")

    # ── Tipboard errors ──
    if "hand_tipboard_err" in f:
        tipboard = f["hand_tipboard_err"][:]  # (T,12) int32
        n_errs = int(np.sum(tipboard != 0))
        if n_errs > 0:
            # Which joints had errors
            err_joints = np.where(np.any(tipboard != 0, axis=0))[0]
            print(f"  tipboard_err: {n_errs} non-zero entries  affected_joints={err_joints.tolist()}")
        else:
            print("  tipboard_err: 全部为 0 (ok)")

    return result


def check_episode(path: str, roi_stride: int, track_thresh_rad: float) -> list[str]:
    """Run all checks on one episode; return the WARN lines."""
    warns: list[str] = []
    with EpisodeReader(path) as reader:
        f = reader.h5f
        meta = dict(f["meta"].attrs) if "meta" in f else {}
        control_hz = float(meta.get("control_hz", 0.0) or 0.0)

        print(f"\n=== {path} ===")
        for k in sorted(meta):
            v = meta[k]
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, float):
                print(f"  meta  {k}={v:.1f}")
            else:
                print(f"  meta  {k}={v}")

        fill_pct = _check_grid(f, control_hz)
        if fill_pct > FILL_WARN_PCT:
            warns.append(f"栅格填充 {fill_pct:.1f}% > {FILL_WARN_PCT:.0f}% — 决策循环跟不上 control_hz")

        cam_keys = [k for k in f.keys() if k == "rgb" or k.endswith("_rgb")]
        # New format: RGB is an MP4 sidecar, not an HDF5 dataset.
        # _MergedH5File.keys() won't include it — try the reader directly.
        if not cam_keys:
            try:
                rgb_data = reader.read_camera_all("rgb")  # decodes MP4 → (T,H,W,3) uint8
                cam_keys.append("rgb")
                # Stash the decoded array so the loop below can find it.
            except KeyError:
                pass
        for key in cam_keys:
            if key == "rgb" and "rgb" not in f:
                data = rgb_data  # pre-decoded numpy array from MP4
            else:
                data = f[key]
            dup_pct, expected_pct = _check_camera(data, key, roi_stride, control_hz)
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

        hand_p95_deg = _check_hand_tracking(f, track_thresh_rad)
        if hand_p95_deg is not None and hand_p95_deg > HAND_TRACK_P95_WARN_DEG:
            warns.append(f"手跟踪误差 p95 {hand_p95_deg:.1f}° > {HAND_TRACK_P95_WARN_DEG:.0f}° — 手跟踪滞后")

        freeze_runs = _check_hand_freeze(f, control_hz)
        if freeze_runs:
            dt = 1.0 / control_hz if control_hz > 0 else float("nan")
            total_frozen = sum(length for _, length, _ in freeze_runs)
            total_frames = int(f["hand_qpos"].shape[0])
            freeze_desc = "  ".join(
                f"{length * dt:.1f}s@t={start * dt:.1f}s(gap={gap:.0f}°)"
                for start, length, gap in freeze_runs[:3]
            )
            print(
                f"  手部冻结: {len(freeze_runs)} 段, 共 {total_frozen}/{total_frames} 帧 "
                f"({100.0 * total_frozen / total_frames:.1f}%)  {freeze_desc}"
            )
            warns.append(
                f"手部冻结 {len(freeze_runs)} 段 / {100.0 * total_frozen / total_frames:.1f}% 帧 — "
                f"电机驱动板锁死，CMD 在执行但 qpos 不动"
            )
        else:
            print("  手部冻结: 无")

        tactile_zero_pcts = _check_tactile(f)
        for finger_name, zero_pct in tactile_zero_pcts.items():
            if zero_pct is not None and zero_pct > TACTILE_ALLZERO_WARN_PCT:
                warns.append(f"触觉 {finger_name} 零值率 {zero_pct:.1f}% > {TACTILE_ALLZERO_WARN_PCT:.0f}% — 传感器可能未初始化(需reset_sensor)或硬件故障")

        # Check tipboard_err for non-zero entries (hardware sensor faults)
        if "hand_tipboard_err" in f:
            tipboard = f["hand_tipboard_err"][:]
            if int(np.sum(tipboard != 0)) > TACTILE_TIPBOARD_ERR_WARN:
                warns.append(f"hand_tipboard_err 存在非零项 — 指尖 PCB 板级传感器故障")

        flags = [k for k in ("flag_ik_ok", "flag_held", "flag_retarget_ok") if k in f]
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
    parser.add_argument("episodes", nargs="+", help="Episode path(s) — .h5 file or directory with data.h5 + depth.h5 + rgb.mp4.")
    parser.add_argument("--roi-stride", type=int, default=8, help="Pixel stride for the rgb content hash (default: 8).")
    parser.add_argument(
        "--track-thresh-rad",
        type=float,
        default=0.35,
        help="Tracking-error threshold in rad for the over-threshold ratio (default: 0.35 = inner-loop warn).",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run trajectory quality assessment (adaptive-threshold classification) alongside health checks.",
    )
    args = parser.parse_args()

    total_warns = 0
    for ep in args.episodes:
        path = Path(ep).expanduser().resolve()
        if not path.exists():
            logger.error("Episode not found: %s", path)
            total_warns += 1
            continue
        try:
            total_warns += len(check_episode(str(path), args.roi_stride, args.track_thresh_rad))
        except (OSError, KeyError, ValueError) as e:
            logger.error("Failed to check %s: %s", path, e)
            total_warns += 1

        # ── Optional quality assessment ──
        if args.quality:
            from dexmani_real.tools.assess_trajectory_quality import assess_episode

            qr = assess_episode(str(path))
            if qr is not None:
                print(
                    f"\n  [quality] {qr.classification:>9s}  "
                    f"{qr.anomaly_ratio*100:.1f}% anomalous  "
                    f"p95={qr.overall_p95_deg:.1f}°  max={qr.overall_max_deg:.1f}°  "
                    f"J{qr.worst_joint} worst"
                )
                if qr.classification == "DEGRADED":
                    total_warns += 1
            else:
                print(f"\n  [quality] SKIP — could not assess")

    if len(args.episodes) > 1:
        print(f"\n共 {len(args.episodes)} 个文件, {total_warns} 项警告")
    sys.exit(1 if total_warns else 0)


if __name__ == "__main__":
    main()

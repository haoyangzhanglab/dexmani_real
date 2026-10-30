#!/usr/bin/env python3
"""不连接设备：重放 Raw VR 关键点或确定性合成序列，评估几何、时延和消融。"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter
from dexmani_real.teleop.retargeting.landmarks import HandCalibration, prepare_landmarks
from dexmani_real.utils.timing import sample_is_fresh


def synthetic_sequence(retargeter, home, count=320):
    """只用于回归/性能，非真实操作者数据，也不模拟接触动力学。"""
    frames = []
    for phase in np.linspace(0, 8 * np.pi, count):
        q = home.copy()
        q[[4, 5, 6, 7, 8, 9, 10, 11]] += (
            0.65 * (1 - np.cos(phase + np.arange(8) * 0.15))
        )
        q[0] += 0.4 * (1 - np.cos(phase))
        q[1] -= 0.3 * (1 - np.cos(phase))
        q[2] += 0.45 * (1 - np.cos(phase))
        q = np.clip(q, retargeter.limits[:, 0], retargeter.limits[:, 1])
        poses = retargeter.fk.forward(q, retargeter.frame_ids)
        points = poses[:, :3, 3].reshape(5, 3, 3)
        human = np.zeros((21, 3))
        human[1], human[2:5] = retargeter.thumb_cmc, points[0]
        for finger in range(1, 5):
            start = 1 + 4 * finger
            human[start:start + 2] = points[finger, :2]
            human[start + 2] = 0.5 * (points[finger, 1] + points[finger, 2])
            human[start + 3] = points[finger, 2]
        frames.append(human)
    return np.array(frames)


def read_episode(path):
    path = path / "data.h5" if path.is_dir() else path
    with h5py.File(path, "r") as data:
        meta = data["meta"].attrs
        for key, expected in (
            ("vr_position_units", "m"), ("vr_coordinate_axes", "FLU"),
            ("vr_hand_side", "right"), ("joint_units", "rad"),
        ):
            if meta.get(key) != expected:
                raise ValueError(f"episode {key} must be {expected}")
        if json.loads(meta["hand_joint_names"]) != list(XHAND_SDK_JOINT_NAMES):
            raise ValueError("episode hand joint order does not match XHand")
        calibration = meta.get("hand_retargeting_calibration_json")
        return (
            data["vr_landmarks"][:], data["timing/rows"]["observation_ns"],
            data["vr_recv_ts_ns"][:], data["vr_sequence_id"][:],
            data["hand_qpos"][0], json.loads(calibration) if calibration else None,
        )


def evaluate(frames, stamps, source_stamps, sequences, initial, runtime, calibration=None):
    count = len(frames)
    if frames.shape != (count, 21, 3) or any(
        values.shape != (count,) for values in (stamps, source_stamps, sequences)
    ):
        raise ValueError("episode landmarks/timestamps/sequences must have matching lengths")
    if count < 2 or np.any(np.diff(stamps.astype(np.int64)) <= 0):
        raise ValueError("evaluation requires >= 2 frames with increasing observation timestamps")
    retargeter = AdaptiveHandRetargeter(
        runtime.hand_retargeting, runtime.hand, control_hz=runtime.teleop.control_hz,
    )
    if calibration is None:
        samples, previous_source, previous_sequence = [], 0, None
        for frame, stamp, source, sequence in zip(frames, stamps, source_stamps, sequences):
            if (
                source > previous_source and sequence != previous_sequence
                and sample_is_fresh(source, runtime.teleop.vr_stale_threshold_s, int(stamp))
            ):
                previous_source, previous_sequence = source, sequence
                prepare_landmarks(frame)
                samples.append(frame)
            if len(samples) == runtime.hand_retargeting.calibration_samples:
                break
        retargeter.calibrate(samples)
        calibration_source = "first_distinct_frames"
    else:
        retargeter.calibration = HandCalibration.from_metadata(calibration)
        calibration_source = "recorded_metadata"
    retargeter.reset(initial, timestamp_ns=int(stamps[0]))
    # 与线上预计算相同，预热求解但不提交、不推进状态。
    retargeter.retarget(frames[0], timestamp_ns=int(stamps[0]))
    elapsed, tips, pinch, velocities = [], [], [], []
    rejected, budgets, reused = 0, 0, 0
    previous_observation, previous_target = None, None
    reasons = {}
    for frame, stamp, source, sequence in zip(frames, stamps, source_stamps, sequences):
        if not sample_is_fresh(source, runtime.teleop.vr_stale_threshold_s, int(stamp)):
            raise ValueError("episode contains stale/future VR input; online control would stop")
        before = retargeter.previous_command.copy()
        started = time.perf_counter()
        # Raw 没有 ring 编号；序号与来源时间共同标识一次接收，支持重连后序号重置。
        observation_id = (int(sequence), int(source))
        if observation_id == previous_observation:
            target = previous_target
            reused += 1
        else:
            target = retargeter.retarget(frame, timestamp_ns=int(stamp))
            previous_observation, previous_target = observation_id, target
            elapsed.append((time.perf_counter() - started) * 1000)
            budgets += int(retargeter.last_diagnostics.get("budget_limited", False))
        if target is None:
            rejected += 1
            reason = retargeter.rejection_reason or "cached rejection"
            reasons[reason] = reasons.get(reason, 0) + 1
            # 线上遇拒绝会终止此段；离线也不继续从旧状态虚构执行。
            break
        points = prepare_landmarks(frame)
        targets, _, activation, distances = retargeter.calibration.targets(
            points, retargeter.robot_points, retargeter.thumb_cmc, retargeter.config,
        )
        actual = retargeter.fk.forward(target, retargeter.frame_ids)[:, :3, 3].reshape(5, 3, 3)
        tips.extend(np.linalg.norm(actual[:, 2] - targets[:, 2], axis=1) * 1000)
        active = activation > 0.5
        gaps = np.linalg.norm(actual[1:, 2] - actual[0, 2], axis=1)
        pinch.extend(np.abs(gaps[active] - distances[active]) * 1000)
        dt = (int(stamp) - retargeter.command_timestamp_ns) * 1e-9 or retargeter.nominal_dt_s
        velocities.extend(np.abs(target - before) / dt)
        retargeter.commit(target, timestamp_ns=int(stamp))
    return dict(
        input_frames=len(frames), completed_frames=len(tips) // 5, rejected=rejected,
        rejection_reasons=reasons, cached_frames=reused, solver_budget_stops=budgets,
        calibration_source=calibration_source, calibration=retargeter.calibration_metadata(),
        latency_ms=dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(elapsed, [50, 95, 99, 100])))),
        tip_target_error_mm=float(np.mean(tips)) if tips else None,
        active_pinch_samples=len(pinch),
        active_pinch_distance_error_mm=float(np.mean(pinch)) if pinch else None,
        max_command_speed_rad_s=float(np.max(velocities)) if velocities else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ablations", action="store_true", help="追加去掉 shape/direction/pinch 的三组比较")
    args = parser.parse_args(argv)
    runtime = load_config(args.config)
    if args.synthetic:
        retargeter = AdaptiveHandRetargeter(
            runtime.hand_retargeting, runtime.hand, control_hz=runtime.teleop.control_hz,
        )
        initial = np.deg2rad(runtime.hand.home_qpos_deg)
        frames = synthetic_sequence(retargeter, initial)
        stamps = 1_000_000_000 + np.rint(np.arange(len(frames)) * 1e9 / runtime.teleop.control_hz).astype(np.int64)
        source_stamps, sequences, calibration = stamps, np.arange(len(frames)), None
    else:
        frames, stamps, source_stamps, sequences, initial, calibration = read_episode(args.episode)
    settings = {"full": runtime}
    if args.ablations:
        for term in ("shape", "direction", "pinch"):
            settings[f"without_{term}"] = replace(
                runtime, hand_retargeting=replace(runtime.hand_retargeting, **{f"{term}_weight": 0.0}),
            )
    results = {
        name: evaluate(frames, stamps, source_stamps, sequences, initial, config, calibration)
        for name, config in settings.items()
    }
    report = dict(
        source="synthetic" if args.synthetic else str(args.episode),
        control_hz=runtime.teleop.control_hz,
        hand_retargeting=asdict(runtime.hand_retargeting),
        operating_limits_rad=[runtime.hand.qpos_min_rad, runtime.hand.qpos_max_rad],
        results=results,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(any(result["rejected"] for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())

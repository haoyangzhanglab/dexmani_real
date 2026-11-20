"""用同一段 Raw 关键点离线比较 SLSQP 和 GeoRT，输出 JSON。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.data.raw import RawReader
from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.commands import validate_hand_target
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.landmarks import (
    FINGER_CHAINS,
    HandCalibration,
    HandGeometryRejected,
    hand_lengths,
    prepare_landmarks,
)
from dexmani_real.teleop.run import build_hand_retargeter

PINCH_FINGERS = ("index", "middle", "ring", "pinky")
PINCH_DISTANCE_M = 0.020
POSITION_TOLERANCE_RAD = 1e-6
VELOCITY_TOLERANCE_RAD_S = 1e-6


def percentiles(values: np.ndarray | list[float], levels: tuple[int, ...]) -> dict[str, float | None]:
    summary = np.percentile(values, levels).tolist() if len(values) else [None] * len(levels)
    return {f"p{level}": value for level, value in zip(levels, summary)}


def thumb_distances(tip_positions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(tip_positions[:, 1:] - tip_positions[:, :1], axis=-1)


def evaluate_backend(retargeter, landmarks, stamps, initial_qpos, near_pinch, fk):
    lower, upper = retargeter.limits.T
    diagnostics = getattr(retargeter, "checkpoint_metadata", None)
    raw_predictions = []
    commands = np.empty((len(landmarks), len(initial_qpos)))
    durations_ms = []
    rejections: Counter[str] = Counter()
    boundary_frames = 0

    retargeter.reset(initial_qpos, timestamp_ns=int(stamps[0]))
    commands[0] = retargeter.reference[0]

    # Raw 的连续区间可以由采样间隙划分。线上间隙后的参考仍是间隙前的已确认命令。
    for row, (points, stamp) in enumerate(zip(landmarks[1:], stamps[1:]), start=1):
        stamp = int(stamp)
        reference, _ = retargeter.reference
        started_ns = perf_counter_ns()
        command = retargeter.retarget(points, timestamp_ns=stamp)
        durations_ms.append((perf_counter_ns() - started_ns) * 1e-6)

        if diagnostics is not None and retargeter.last_prediction_rad is not None:
            raw_predictions.append(retargeter.last_prediction_rad.copy())
        if command is None:
            rejections[retargeter.rejection_reason] += 1
            command = reference
        else:
            command = validate_hand_target(command, lower, upper)
            distance_to_limit = np.minimum(command - lower, upper - command)
            boundary_frames += int(np.any(distance_to_limit <= POSITION_TOLERANCE_RAD))

        commands[row] = command
        retargeter.commit(command, timestamp_ns=stamp)

    robot_tips = np.stack([fk.forward(qpos, fk.fingertip_frame_ids)[:, :3, 3] for qpos in commands])
    robot_distances_mm = 1000 * thumb_distances(robot_tips)
    speeds = np.abs(np.diff(commands, axis=0)) / (np.diff(stamps)[:, None] * 1e-9)
    peak_speeds = speeds.max(axis=1)
    joint_p95 = np.percentile(speeds, 95, axis=0).tolist() if len(speeds) else [None] * speeds.shape[1]
    speed_limit = retargeter.config.max_joint_speed_rad_s
    rejected_frames = sum(rejections.values())
    report = {
        "accepted_frames": len(durations_ms) - rejected_frames,
        "rejected_frames": rejected_frames,
        "rejection_reasons": dict(rejections),
        "retarget_ms": percentiles(durations_ms, (50, 95, 99, 100)),
        "command_frames_near_position_limit": boundary_frames,
        "command_motion_rad_s": {
            "frame_peak": percentiles(peak_speeds, (50, 95, 100)),
            "p95_by_joint": dict(zip(XHAND_SDK_JOINT_NAMES, joint_p95)),
        },
        "command_frames_exceeding_speed_limit": int(
            np.count_nonzero(peak_speeds > speed_limit + VELOCITY_TOLERANCE_RAD_S)
        ),
        "robot_pinch_distance_mm": {
            finger: percentiles(robot_distances_mm[near_pinch[:, i], i], (50, 95))
            for i, finger in enumerate(PINCH_FINGERS)
        },
    }
    if diagnostics is not None:
        predictions = np.asarray(raw_predictions).reshape(-1, len(XHAND_SDK_JOINT_NAMES))
        below = predictions < lower
        above = predictions > upper
        report["checkpoint"] = diagnostics
        report["raw_prediction"] = {
            "frames": len(predictions),
            "frames_outside_position_limits": int(np.count_nonzero(np.any(below | above, axis=1))),
            "max_position_excess_rad": float(np.maximum(lower - predictions, predictions - upper).max(initial=0)),
            "below_min_by_joint": dict(zip(XHAND_SDK_JOINT_NAMES, below.sum(axis=0).tolist())),
            "above_max_by_joint": dict(zip(XHAND_SDK_JOINT_NAMES, above.sum(axis=0).tolist())),
        }
    return report, commands


def write_trajectory(path: Path, trajectories: dict[str, np.ndarray], stamps: np.ndarray, fk: HandFK) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    from dexmani_real.config.perception import HandCloudConfig
    from dexmani_real.perception.hand_cloud import HandPointCloud
    from dexmani_real.utils.visualization.rerun import log_points

    cloud = HandPointCloud(fk, HandCloudConfig().num_points)
    blueprint = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="/slsqp", name="SLSQP"),
        rrb.Spatial3DView(origin="/geort", name="GeoRT"),
    ))
    rr.init("XHand Retargeting", strict=True)
    rr.save(path, default_blueprint=blueprint)
    try:
        for row, stamp in enumerate(stamps - stamps[0]):
            rr.set_time_sequence("step", row)
            rr.set_time_nanos("time", int(stamp))
            for backend, commands in trajectories.items():
                placement = fk.forward(commands[row], cloud.frame_ids)
                points = cloud.transform_batch(placement[None])[0]
                log_points(f"{backend}/hand", points, colors=[100, 180, 255], radii=0.0015)
    finally:
        rr.disconnect()
    if not path.is_file():
        raise RuntimeError(f"Rerun 未生成轨迹文件（{path}）")


def evaluate_episode(
    path: Path, checkpoint_dir: Path, fk: HandFK, trajectory_output: Path | None = None,
) -> dict:
    with RawReader(path) as reader:
        if reader.meta.get("collection_source") != "teleop" or reader.num_frames == 0:
            raise ValueError(f"需要带有 HTS 关键点的非空 VR 遥操作录制段（{reader.path}）")
        parameters = json.loads(reader.meta["effective_parameters"])
        settings = HandRetargetingConfig(**parameters["hand_retargeting"])
        hand = HandConfig(
            qpos_min_rad=tuple(parameters["hand"]["qpos_min_rad"]),
            qpos_max_rad=tuple(parameters["hand"]["qpos_max_rad"]),
        )
        episode_path = reader.path
        control_hz = reader.control_hz
        landmarks = reader["vr_landmarks"][:]
        initial_qpos = reader["hand_qpos"][0]
        stamps = reader.read_timing()["observation_ns"]
        segment_count = len(reader.segment_starts)

    hand.validate_limits()
    calibration = HandCalibration.from_recording(parameters["hand_retargeting_calibration"])

    near_pinch = thumb_distances(landmarks[:, FINGER_CHAINS[:, -1]]) < PINCH_DISTANCE_M
    for row in np.flatnonzero(near_pinch.any(axis=1)):
        try:
            lengths = hand_lengths(prepare_landmarks(landmarks[row]))
            calibration.validate_lengths(lengths, tolerance=settings.bone_length_tolerance)
        except HandGeometryRejected:
            near_pinch[row] = False
    report = {
        "episode": str(episode_path),
        "control_hz": control_hz,
        "frames": len(landmarks),
        "segments": segment_count,
        "command_speed_limit_rad_s": settings.max_joint_speed_rad_s,
        "human_pinch_distance_threshold_mm": 1000 * PINCH_DISTANCE_M,
        "human_pinch_frames": dict(zip(PINCH_FINGERS, near_pinch.sum(axis=0).tolist())),
    }
    trajectories = {}
    for backend, model_dir in (("slsqp", None), ("geort", str(checkpoint_dir))):
        config = replace(settings, backend=backend, geort_checkpoint_dir=model_dir)
        retargeter = build_hand_retargeter(RealConfig(hand=hand, hand_retargeting=config))
        retargeter.calibration = calibration
        report[backend], trajectories[backend] = evaluate_backend(
            retargeter, landmarks, stamps, initial_qpos, near_pinch, fk,
        )
    if trajectory_output is not None:
        write_trajectory(trajectory_output, trajectories, stamps, fk)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="带 HTS 数据的 Raw episode 目录")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="GeoRT checkpoint 目录")
    parser.add_argument("--trajectory-output", type=Path, help="将单个 episode 的 A/B 手形保存为 Rerun .rrd")
    args = parser.parse_args(argv)
    if args.trajectory_output is not None:
        if len(args.episodes) != 1 or args.trajectory_output.suffix != ".rrd":
            parser.error("--trajectory-output 需要单个 Raw episode 和 .rrd 输出路径")
        if args.trajectory_output.exists():
            parser.error(f"--trajectory-output 的输出文件已存在（{args.trajectory_output}）")

    fk = HandFK()
    reports = [evaluate_episode(path, args.checkpoint_dir, fk, args.trajectory_output) for path in args.episodes]
    print(json.dumps({
        "timing": "每份 Raw 首帧用于实测姿态初始化。采样间隙沿用上一确认参考。"
        "使用观测时刻计算并模拟确认提交。拒绝帧模拟在当前时刻确认保持上一目标。"
        "离线计算省略实际设备和租约。",
        "pinch_metric": "仅用通过人手几何校验且人手尖距低于阈值的样本，计算同帧 XHand FK 指尖距。",
        "episodes": reports,
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return int(any(
        report[backend]["command_frames_exceeding_speed_limit"]
        for report in reports for backend in ("slsqp", "geort")
    ))


if __name__ == "__main__":
    raise SystemExit(main())

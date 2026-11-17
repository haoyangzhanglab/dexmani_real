"""从 Raw 记录独立计算手部重定向，并在手部局部坐标系预览。"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.data.raw import RawReader
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.landmarks import HandCalibration
from dexmani_real.teleop.retargeting.solver import HandRetargeter


def read_recorded_parameters(reader: RawReader, config_path: str | Path | None = None):
    parameters = json.loads(reader.meta["effective_parameters"])
    if config_path is None:
        recorded = parameters["hand_retargeting"]
        missing = {field.name for field in fields(HandRetargetingConfig)} - recorded.keys()
        if missing:
            raise ValueError(f"Raw hand_retargeting missing parameters: {', '.join(sorted(missing))}")
        config = HandRetargetingConfig(**recorded)
    else:
        config = load_config(config_path).hand_retargeting
    hand = parameters["hand"]
    limits = HandConfig(qpos_min_rad=hand["qpos_min_rad"], qpos_max_rad=hand["qpos_max_rad"])
    limits.validate_limits()
    recorded_calibration = parameters["hand_retargeting_calibration"]
    bones = np.asarray(recorded_calibration["bone_lengths"], dtype=np.float64)
    width = np.asarray(recorded_calibration["palm_width"], dtype=np.float64)
    if bones.shape != (5, 3) or not np.isfinite(bones).all() or np.any(bones <= 0):
        raise ValueError("Raw hand_retargeting_calibration.bone_lengths must be positive finite (5, 3)")
    if width.shape != () or not np.isfinite(width) or width <= 0:
        raise ValueError("Raw hand_retargeting_calibration.palm_width must be a positive finite scalar")
    return config, limits, HandCalibration(bones, float(width))


def preview_episode(
    episode_path: str | Path, *, config_path: str | Path | None = None,
    max_frames: int | None = None, view: bool = True,
) -> dict:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    with RawReader(episode_path) as reader:
        count = reader.num_frames if max_frames is None else min(reader.num_frames, max_frames)
        if count == 0:
            raise ValueError("episode must contain at least one row")
        config, limits, calibration = read_recorded_parameters(reader, config_path)
        timestamps = reader.read_timing(stop=count)["observation_ns"]
        starts = {int(start) for start in reader.segment_starts if start < count}
        measured = reader["hand_qpos"][:count]
        landmarks = reader["vr_landmarks"][:count]
        if not np.isfinite(measured).all():
            raise ValueError("Raw hand_qpos contains nonfinite values in the selected prefix")
        episode_name = reader.path.name

    retargeter = HandRetargeter(config, limits)
    retargeter.calibration = calibration
    candidates = np.full(measured.shape, np.nan)
    references = np.empty_like(candidates)
    solve_times = []
    reasons = [None] * count
    for index, timestamp in enumerate(timestamps):
        timestamp = int(timestamp)
        if index in starts:
            retargeter.reset(measured[index], timestamp_ns=timestamp)
        else:
            if not np.isfinite(landmarks[index]).all():
                raise ValueError(f"Raw vr_landmarks contains nonfinite values at row {index}")
            before = perf_counter()
            candidate = retargeter.retarget(landmarks[index], timestamp_ns=timestamp)
            solve_times.append(perf_counter() - before)
            if candidate is None:
                reasons[index] = retargeter.rejection_reason
            else:
                candidates[index] = candidate
                # 离线接受候选；这里不表示真实设备曾成功提交该命令。
                retargeter.commit(candidate, timestamp_ns=timestamp)
        references[index] = retargeter.reference_qpos

    result = {
        "observation_ns": timestamps, "segment_starts": starts,
        "measured_qpos": measured, "candidate_qpos": candidates, "reference_qpos": references,
        "rejection_reasons": reasons, "solve_times_s": np.asarray(solve_times),
    }
    if view:
        show_preview(result, retargeter.fk, episode_name)
    return result


def show_preview(result: dict, fk, episode_name: str) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    from dexmani_real.utils.visualization.rerun import log_points, log_series

    rr.init(f"Hand Retargeting - {episode_name}", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/hand", name="Hand local frame (m)"),
            rrb.TextLogView(origin="/events", name="Segment starts and rejections"),
        ),
        rrb.Vertical(
            rrb.Tabs(contents=[
                rrb.TimeSeriesView(
                    origin="/qpos", name=joint,
                    contents=[
                        f"/qpos/{kind}/{index}" for kind in ("measured", "candidate", "reference")
                    ],
                )
                for index, joint in enumerate(XHAND_SDK_JOINT_NAMES)
            ]),
            rrb.TimeSeriesView(origin="/status", name="Segment starts and rejected rows"),
        ),
    )))
    timestamps = result["observation_ns"]
    # 仅显示轴转换为相对秒；求解器始终收到原始 observation_ns。
    seconds = (timestamps - timestamps[0]) / 1e9
    steps = np.arange(len(timestamps))
    for kind in ("measured", "candidate", "reference"):
        log_series(
            f"qpos/{kind}", result[f"{kind}_qpos"], steps=steps, timestamps_s=seconds,
            labels=[f"{kind} (rad)"] * len(XHAND_SDK_JOINT_NAMES),
        )
    starts = result["segment_starts"]
    for index, second in enumerate(seconds):
        rr.set_time_sequence("step", index)
        rr.set_time_seconds("time", float(second))
        reason = result["rejection_reasons"][index]
        is_start = index in starts
        is_rejected = reason is not None
        rr.log("status/segment_start", rr.Scalar(int(is_start)))
        rr.log("status/rejected", rr.Scalar(int(is_rejected)))
        if is_start:
            rr.log("events", rr.TextLog(f"row {index}: reset to measured pose; reference only"))
        elif is_rejected:
            rr.log("events", rr.TextLog(f"row {index}: rejected: {reason}", level="WARN"))
        # 成功候选已成为参考，避免重叠绘制相同的指尖。
        target_kind = "reference" if is_start or is_rejected else "candidate"
        for kind, color in (
            ("measured", [70, 160, 255]), ("candidate", [255, 170, 50]), ("reference", [170, 170, 170]),
        ):
            points = None
            if kind == "measured" or kind == target_kind:
                qpos = result[f"{kind}_qpos"][index]
                points = fk.forward(qpos, fk.fingertip_frame_ids)[:, :3, 3]
            log_points(f"hand/{kind}/fingertips", points, colors=color, radii=0.004)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", metavar="EPISODE", help="Raw 目录；相对 cwd，支持 ~")
    parser.add_argument(
        "--config",
        help="导出 CONFIG: RealConfig 的 Python 文件；仅替换 hand_retargeting，标定和限位仍取自 Raw",
    )
    parser.add_argument("--max-frames", type=int, help="只读取并计算前 N 行")
    parser.add_argument("--no-view", action="store_true", help="仅计算并打印统计，不初始化 Rerun")
    args = parser.parse_args(argv)
    result = preview_episode(
        args.episode, config_path=args.config, max_frames=args.max_frames, view=not args.no_view,
    )
    rejected = [
        (index, reason) for index, reason in enumerate(result["rejection_reasons"])
        if reason is not None
    ]
    print(f"Frames: {len(result['observation_ns'])}; rejected: {len(rejected)}")
    for index, reason in rejected:
        print(f"  Rejected row {index}: {reason}")
    times_ms = result["solve_times_s"] * 1000
    if len(times_ms):
        print(
            f"Solve time ({len(times_ms)} calls, ms): mean={times_ms.mean():.3f}, "
            f"p95={np.percentile(times_ms, 95):.3f}, max={times_ms.max():.3f}"
        )
    else:
        print("Solve time: no samples (only segment reference rows selected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

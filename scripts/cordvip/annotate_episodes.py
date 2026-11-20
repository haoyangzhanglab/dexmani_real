"""
离线跟踪已发布 Raw，短暂缺帧时保持最近一次完整场景。
python scripts/cordvip/annotate_episodes.py \
  episodes/pnp_apple_banana/episode_20261120_193502 \
  --task-dir assets/pointcloud_template/object/pnp_apple_banana
"""

import argparse
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry import RGBDFrame, SceneTracker

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HOITrackingConfig
from dexmani_real.config.real import load_config
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import (
    CAMERA_TIMING_DTYPE,
    CAMERA_TIMING_FIELDS,
    RawDataError,
    RawReader,
    discover_episode_dirs,
)
from dexmani_real.geometry.camera import CameraSnapshot
from dexmani_real.geometry.hoi import load_hoi_task
from dexmani_real.perception.hoi import build_tracking_config, track_hoi_frame
from dexmani_real.utils.timing import sample_is_fresh

GAP_WINDOW_NS = 1_000_000_000


def annotate_episodes(
    episodes: Iterable[str | Path],
    task: str | Path,
    *,
    tracking_config: HOITrackingConfig,
    overwrite: bool = False,
    repair_gaps: bool = False,
    hoi_max_age_s: float = ExecutionConfig().hoi_max_age_s,
) -> None:
    if type(hoi_max_age_s) not in (int, float) or not math.isfinite(hoi_max_age_s) or hoi_max_age_s <= 0:
        raise ValueError("hoi_max_age_s requires a finite positive value")
    config = build_tracking_config(tracking_config)
    tracker = None

    for episode in episodes:
        with RawReader(episode) as reader:
            output = reader.path / "hoi.h5"
            if output.exists() and not overwrite:
                _ = reader.hoi_file
                print(f"跳过已有 HOI：{reader.path}")
                continue
            if reader.num_frames == 0:
                raise RawDataError("空 Episode")
            timing = reader.read_timing(source_clocks=("camera_ns",))
            segment_starts = reader.segment_starts
            camera = reader.read_camera()
            base_from_camera = camera.T_xarm_base_from_color
            if base_from_camera is None:
                raise RawDataError(f"Raw {reader.path}: HOI 必须有记录的相机外参")
            depth = reader["depth"]
            if tracker is None:
                template, mesh_paths, prompts = load_hoi_task(Path(task).expanduser().resolve())
                tracker = SceneTracker(mesh_paths, workspace=ROOT, config=config)
            poses, quality, valid = track_episode(
                reader, tracker, template, prompts, timing, camera=camera, depth=depth,
            )
            source_fresh = np.fromiter(
                (sample_is_fresh(stamp, hoi_max_age_s, now)
                 for stamp, now in zip(timing["camera_ns"], timing["observation_ns"])),
                dtype=bool, count=len(timing),
            )
            observed = valid.all(axis=1)
            settings = asdict(config)
            if repair_gaps:
                repaired = repair_short_gaps(
                    reader, tracker, template["object_names"], poses, timing, quality, valid, source_fresh,
                    camera=camera, depth=depth, segment_starts=segment_starts,
                )
            held_rows = hold_recent_scene(
                segment_starts, poses, timing, quality, valid, observed, source_fresh, hoi_max_age_s,
            )
            if repair_gaps:
                if held_rows and repaired:
                    retained_rows = source_fresh.copy()
                    retained_rows[list(held_rows)] = False
                    retained_sequences = set(timing["camera_sequence"][retained_rows])
                    # 同一相机序列的重复行可能只在部分行被整场景保持覆盖。
                    for name in tuple(repaired):
                        sequences = [seq for seq in repaired[name] if seq in retained_sequences]
                        if sequences:
                            repaired[name] = sequences
                        else:
                            del repaired[name]
                settings["offline_backtrack"] = repaired
                print(f"{reader.path}：补回 {sum(map(len, repaired.values()))} 个物体位姿")
            settings["short_hold"] = {"max_age_s": hoi_max_age_s, "held_rows": held_rows}
            if held_rows:
                print(f"{reader.path}：短时保持 {len(held_rows)} 行物体点云")
            provisional_rows = {}
            for row, row_quality in enumerate(quality):
                objects = [
                    str(name) for name, metrics in zip(template["object_names"], row_quality, strict=True)
                    if metrics.get("provisional")
                ]
                if objects:
                    provisional_rows[row] = objects
                    quality[row] = tuple(
                        {key: value for key, value in metrics.items() if key != "provisional"}
                        for metrics in row_quality
                    )
            if provisional_rows:
                settings["provisional_rows"] = provisional_rows
                print(f"{reader.path}：使用当帧备用位姿 {len(provisional_rows)} 行")

        output = write_hoi(
            output,
            template,
            poses,
            base_from_camera,
            camera_timing=timing[list(CAMERA_TIMING_FIELDS)].astype(CAMERA_TIMING_DTYPE),
            quality=quality,
            object_valid=valid,
            tracking_config=settings,
            overwrite=overwrite,
        )
        print(f"{output} 已写入 {len(poses)} 帧")


def track_episode(
    reader: RawReader,
    tracker: SceneTracker,
    template: Mapping[str, np.ndarray],
    prompts: dict[str, str],
    timing: np.ndarray,
    *,
    camera: CameraSnapshot,
    depth,
):
    frames = len(timing)
    sequences, stamps = timing["camera_sequence"], timing["camera_ns"]
    new_sequence = np.r_[True, sequences[1:] != sequences[:-1]]

    objects = len(template["object_names"])
    poses = np.empty((frames, objects, 4, 4), np.float32)
    valid = np.empty((frames, objects), bool)
    initialize = True
    quality: list[tuple[dict[str, float | int | None], ...]] = []
    for index, rgb in enumerate(reader.iter_rgb_frames()):
        if not new_sequence[index]:
            poses[index] = poses[index - 1]
            quality.append(quality[-1])
            valid[index] = valid[index - 1]
            continue
        frame = RGBDFrame.from_camera(
            camera,
            {"rgb": rgb, "depth": depth[index], "timestamp_ns": stamps[index]},
            source_sequence=sequences[index],
        )
        result = track_hoi_frame(tracker, frame, template, prompts, initialize=initialize)
        initialize = initialize and not result.valid
        if initialize:
            reason = next(iter(result.failures.values()))
            print(f"行 {index}，相机序列 {sequences[index]}: {reason}；继续重试")
        poses[index] = result.object_pose_camera
        valid[index] = result.object_valid
        quality.append(result.quality)

    return poses, quality, valid


def hold_recent_scene(segment_starts, poses, timing, quality, valid, observed, source_fresh, max_age_s) -> dict[int, int]:
    """检查当前来源的时效，缺失时暂时沿用同段已观测过的完整场景。"""
    sequences, stamps = timing["camera_sequence"], timing["camera_ns"]
    observation_ns = timing["observation_ns"]
    segment_start_rows = set(segment_starts)
    no_quality: tuple[dict[str, float | int | None], ...] = tuple({} for _ in range(valid.shape[1]))
    held_rows: dict[int, int] = {}
    source = None
    for row in range(len(poses)):
        if row in segment_start_rows:
            source = None
        # 同一相机帧可对应多个控制行，来源时间戳始终相同。
        if not source_fresh[row]:
            valid[row] = False
            quality[row] = no_quality
            continue
        if observed[row]:
            # 段首重复的相机帧仍属于上一段。
            if row == 0 or sequences[row] != sequences[row - 1]:
                source = row
        elif (
            not valid[row].all() and source is not None
            and sample_is_fresh(stamps[source], max_age_s, observation_ns[row])
        ):
            poses[row] = poses[source]
            valid[row] = True
            quality[row] = no_quality
            held_rows[row] = int(sequences[source])
    return held_rows


def repair_short_gaps(
    reader, tracker, names, poses, timing, quality, valid, source_fresh, *, camera, depth, segment_starts,
) -> dict[str, list[int]]:
    """只回补同一 Raw 段内、由两侧有效观测夹住的短缺口。"""
    sequence, times = timing["camera_sequence"], timing["camera_ns"]
    rows = np.flatnonzero(np.r_[True, sequence[1:] != sequence[:-1]])
    segment = np.searchsorted(segment_starts, rows, side="right")
    trusted = valid & source_fresh[:, None] & np.array([
        [not obj.get("provisional") for obj in row] for row in quality
    ])
    gaps: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    needed: set[int] = set()
    for obj in range(len(names)):
        anchors = np.flatnonzero(trusted[rows, obj])
        for left, right in pairwise(anchors):
            before, after = int(rows[left]), int(rows[right])
            if (
                right > left + 1 and segment[left] == segment[right]
                and int(times[after]) - int(times[before]) <= GAP_WINDOW_NS
            ):
                missing = rows[left + 1 : right]
                eligible = missing[source_fresh[missing]]
                if not eligible.size:
                    continue
                gaps.setdefault(after, []).append((obj, before, missing))
                needed.update(eligible)
    repaired: dict[str, list[int]] = {}
    if not gaps:
        return repaired

    row_end = dict(zip(rows, np.r_[rows[1:], len(poses)], strict=True))
    cached = {}
    last_anchor = max(gaps)
    for after, rgb in enumerate(reader.iter_rgb_frames()):
        if after in needed:
            cached[after] = RGBDFrame.from_camera(
                camera, {"rgb": rgb, "depth": depth[after], "timestamp_ns": times[after]},
                source_sequence=sequence[after],
            )
        if after not in gaps:
            continue
        anchor_ns = int(times[after])
        cached = {
            row: frame for row, frame in cached.items()
            if anchor_ns - int(frame.timestamp_ns) <= GAP_WINDOW_NS
        }
        for obj, before, missing in gaps[after]:
            name = str(names[obj])
            steps = list(zip(
                missing, np.r_[before, missing[:-1]], np.r_[missing[1:], after], strict=True
            ))
            for anchor, path in ((after, reversed(steps)), (before, steps)):
                seed, seed_ns = poses[anchor, obj], int(times[anchor])
                for row, left, right in path:
                    if trusted[row, obj] or not source_fresh[row]:
                        break
                    result = tracker.refine_from_pose(name, cached[row], seed, seed_ns)
                    if result is None:
                        break
                    candidate, metrics = result
                    # 深度可能无法区分朝向，接回相邻有效帧时拒绝超过 90° 的翻转。
                    turns = (
                        np.trace(poses[neighbor, obj, :3, :3].T @ candidate[:3, :3])
                        for neighbor in (left, right) if trusted[neighbor, obj]
                    )
                    if any(turn < 1 for turn in turns):
                        break
                    stop = row_end[row]
                    span = slice(row, stop)
                    poses[span, obj] = candidate
                    valid[span, obj] = True
                    trusted[span, obj] = True
                    row_quality = quality[row]
                    quality[span] = [row_quality[:obj] + (metrics,) + row_quality[obj + 1 :]] * (stop - row)
                    repaired.setdefault(name, []).append(int(sequence[row]))
                    if anchor == after:
                        seed, seed_ns = candidate, int(times[row])
        if after == last_anchor:
            break
    return repaired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "input_root",
        metavar="INPUT_ROOT",
        type=Path,
        help="一个 Episode，或直接包含 episode_* 子目录的数据目录；相对 cwd，支持 ~",
    )
    parser.add_argument("--task-dir", type=Path, required=True, help="任务资产目录；相对 cwd，支持 ~")
    parser.add_argument("--config", type=Path, help="导出 CONFIG: RealConfig 的可信 Python 配置")
    parser.add_argument("--overwrite", action="store_true", help="显式重算并原子替换已有 hoi.h5")
    parser.add_argument("--repair-gaps", action="store_true", help="离线回补有同段观测锚点的短位姿缺口")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    annotate_episodes(
        discover_episode_dirs(args.input_root),
        args.task_dir,
        overwrite=args.overwrite,
        repair_gaps=args.repair_gaps,
        tracking_config=config.hoi_tracking,
        hoi_max_age_s=config.execution.hoi_max_age_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""离线跟踪已发布 Raw，保留暂时丢失帧的有效性并继续恢复。"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.camera import RGBDFrame
from cordvip_geometry.tracking import SceneTracker, TrackingConfig, TrackingFailure

from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import (
    CAMERA_TIMING_DTYPE,
    CAMERA_TIMING_FIELDS,
    RawReader,
    discover_episode_dirs,
)
from dexmani_real.geometry.hoi import load_hoi_task


def annotate_episodes(
    episodes, task, *,
    tracking_config: TrackingConfig = TrackingConfig(),  # noqa: B008 — 不可变配置可共享。
    overwrite=False,
):
    task = Path(task).expanduser().resolve()
    tracker = None

    for episode in episodes:
        episode = Path(episode).expanduser()
        with RawReader(episode) as reader:
            if (episode / "hoi.h5").exists() and not overwrite:
                _ = reader.hoi_file
                print(f"跳过已有 HOI：{episode}")
                continue
            if tracker is None:
                template, mesh_paths, prompts = load_hoi_task(task)
                names = template["object_names"]
                tracker = SceneTracker(
                    mesh_paths, workspace=ROOT, config=tracking_config
                )
            poses, base_from_camera, camera_timing, quality, valid = track_episode(
                reader, tracker, names, prompts
            )

        write_hoi(
            episode / "hoi.h5",
            template,
            poses,
            base_from_camera,
            camera_timing=camera_timing,
            quality=quality,
            object_valid=valid,
            tracking_config=asdict(tracker.config),
            overwrite=overwrite,
        )
        print(f"{episode}: 已写入 HOI，{len(poses)} 帧")


def track_episode(reader, tracker, names, prompts):
    camera = reader.read_camera()
    if camera.T_xarm_base_from_color is None:
        raise ValueError("HOI 必须有 Raw 中记录的相机外参")

    frames = reader.num_frames
    if frames == 0:
        raise ValueError("空 Episode")
    depth = reader["depth"]

    timing = reader.read_timing(source_clocks=("camera_ns",))
    camera_timing = timing[list(CAMERA_TIMING_FIELDS)].astype(CAMERA_TIMING_DTYPE)
    sequences, stamps = camera_timing["camera_sequence"], camera_timing["camera_ns"]
    new_sequence = np.r_[True, sequences[1:] != sequences[:-1]]
    invalid_sequence = sequences <= 0
    invalid_sequence[1:] |= sequences[1:] < sequences[:-1]
    time_conflict = np.r_[False, ~new_sequence[1:] & (stamps[1:] != stamps[:-1])]
    bad_rows = np.flatnonzero(invalid_sequence | time_conflict)
    if bad_rows.size:
        index = bad_rows[0]
        reason = (
            "非法相机序列" if invalid_sequence[index] else "相同相机序列的来源时间冲突"
        )
        raise ValueError(f"{reason}，行 {index}")

    poses = np.full((frames, len(names), 4, 4), np.nan, np.float32)
    valid = np.zeros((frames, len(names)), bool)
    first = True
    quality: list[list[dict[str, float | int | None]]] = []
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
        try:
            result = (
                tracker.initialize(frame, prompts) if first else tracker.track(frame)
            )
        except TrackingFailure as error:
            print(f"行 {index}，相机序列 {sequences[index]}: {error}；继续重试")
            quality.append([{} for name in names])
            continue
        first = False
        for obj, name in enumerate(names):
            valid[index, obj] = result.valid[name]
            if result.valid[name]:
                poses[index, obj] = result.poses[name]
        quality.append([result.quality[name] for name in names])

    return poses, camera.T_xarm_base_from_color, camera_timing, quality, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("input_root", metavar="INPUT_ROOT", type=Path,
                        help="一个 Episode，或直接包含 episode_* 子目录的数据目录；相对 cwd，支持 ~")
    parser.add_argument("--task-dir", type=Path, required=True, help="任务资产目录；相对 cwd，支持 ~")
    parser.add_argument("--overwrite", action="store_true", help="显式重算并原子替换已有 hoi.h5")
    parser.add_argument(
        "--segmentation-interval-frames",
        type=int,
        default=TrackingConfig().segmentation_interval_frames,
    )
    args = parser.parse_args(argv)
    annotate_episodes(
        discover_episode_dirs(args.input_root),
        args.task_dir,
        overwrite=args.overwrite,
        tracking_config=TrackingConfig(
            segmentation_interval_frames=args.segmentation_interval_frames
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

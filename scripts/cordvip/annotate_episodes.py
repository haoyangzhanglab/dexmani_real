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
from dexmani_real.data.raw import RawReader, discover_episode_dirs
from dexmani_real.geometry.hoi import load_task_template, task_objects


def annotate_episodes(
    episodes, task, *,
    tracking_config: TrackingConfig = TrackingConfig(),  # noqa: B008 — 不可变配置可共享。
    overwrite=False,
):
    task = Path(task).expanduser().resolve()
    tracker = None

    for episode in episodes:
        episode = Path(episode).expanduser()
        if (episode / "hoi.h5").exists() and not overwrite:
            print(f"跳过已有 HOI：{episode}")
            continue
        with RawReader(episode) as reader:
            if tracker is None:
                template = load_task_template(task)
                names = template["object_names"]
                by_name = {obj["name"]: obj for obj in task_objects(task)}
                mesh_paths = {name: task / name / "mesh/mesh_m.obj" for name in names}
                prompts = {name: by_name[name]["prompt"] for name in names}
                tracker = SceneTracker(
                    mesh_paths, workspace=ROOT, config=tracking_config
                )
            poses, base_from_camera, quality, valid = track_episode(
                reader, tracker, names, prompts
            )

        write_hoi(
            episode / "hoi.h5",
            template,
            poses,
            base_from_camera,
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
    if camera.depth_scale_m is None:
        raise ValueError("HOI 必须有 Raw 中记录的深度单位")

    frames = reader.num_frames
    if frames == 0:
        raise ValueError("空 Episode")
    reader.require_fields("rgb", "depth")
    depth = reader["depth"]

    timing = reader["timing/rows"][:]
    sequences, stamps = timing["camera_sequence"], timing["camera_ns"]
    new_sequence = np.ones(frames, dtype=bool)
    new_sequence[1:] = sequences[1:] != sequences[:-1]
    invalid_clock = (sequences <= 0) | (stamps <= 0)
    invalid_clock[1:] |= (sequences[1:] < sequences[:-1]) | (stamps[1:] < stamps[:-1])
    time_conflict = np.zeros(frames, dtype=bool)
    time_conflict[1:] = ~new_sequence[1:] & (stamps[1:] != stamps[:-1])
    bad_rows = np.flatnonzero(invalid_clock | time_conflict)
    if bad_rows.size:
        index = bad_rows[0]
        reason = (
            "非法相机时钟" if invalid_clock[index] else "相同相机序列的来源时间冲突"
        )
        raise ValueError(f"{reason}，行 {index}")

    poses = np.full((frames, len(names), 4, 4), np.nan, np.float32)
    valid = np.zeros((frames, len(names)), bool)
    first = True
    quality: list[list[dict[str, float | int | None]]] = []
    images = iter(reader.iter_rgb_frames())
    for index in range(frames):
        rgb = next(images, None)
        if rgb is None:
            raise ValueError(f"Raw RGB ended before row {index}")
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

    if next(images, None) is not None:
        raise ValueError("Raw RGB contains more frames than the HDF5 rows")
    return poses, camera.T_xarm_base_from_color, quality, valid


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

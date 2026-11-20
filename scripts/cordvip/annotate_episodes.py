"""离线跟踪已发布 Raw，保留暂时丢失帧的有效性并继续恢复。"""

import argparse
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry import RGBDFrame, SceneTracker

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
from dexmani_real.geometry.hoi import load_hoi_task
from dexmani_real.perception.hoi import build_tracking_config, track_hoi_frame


def annotate_episodes(
    episodes: Iterable[str | Path],
    task: str | Path,
    *,
    tracking_config: HOITrackingConfig,
    overwrite: bool = False,
) -> None:
    config = build_tracking_config(tracking_config)
    tracker = None

    for episode in episodes:
        with RawReader(episode) as reader:
            output = reader.path / "hoi.h5"
            if output.exists() and not overwrite:
                _ = reader.hoi_file
                print(f"跳过已有 HOI：{reader.path}")
                continue
            if tracker is None:
                template, mesh_paths, prompts = load_hoi_task(Path(task).expanduser().resolve())
                tracker = SceneTracker(mesh_paths, workspace=ROOT, config=config)
            poses, base_from_camera, camera_timing, quality, valid = track_episode(
                reader, tracker, template, prompts
            )

        output = write_hoi(
            output,
            template,
            poses,
            base_from_camera,
            camera_timing=camera_timing,
            quality=quality,
            object_valid=valid,
            tracking_config=asdict(config),
            overwrite=overwrite,
        )
        print(f"{output} 已写入 {len(poses)} 帧")


def track_episode(
    reader: RawReader,
    tracker: SceneTracker,
    template: Mapping[str, np.ndarray],
    prompts: dict[str, str],
):
    camera = reader.read_camera()
    if camera.T_xarm_base_from_color is None:
        raise RawDataError("HOI 必须有 Raw 中记录的相机外参")

    frames = reader.num_frames
    if frames == 0:
        raise RawDataError("空 Episode")
    depth = reader["depth"]

    timing = reader.read_timing(source_clocks=("camera_ns",))
    camera_timing = timing[list(CAMERA_TIMING_FIELDS)].astype(CAMERA_TIMING_DTYPE)
    sequences, stamps = camera_timing["camera_sequence"], camera_timing["camera_ns"]
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

    return poses, camera.T_xarm_base_from_color, camera_timing, quality, valid


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
    args = parser.parse_args(argv)
    annotate_episodes(
        discover_episode_dirs(args.input_root),
        args.task_dir,
        overwrite=args.overwrite,
        tracking_config=load_config(args.config).hoi_tracking,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

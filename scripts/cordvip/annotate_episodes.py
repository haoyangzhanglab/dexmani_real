"""将已发布 Raw 的离线 HOI 结果写入原 data.h5。"""

# Usage: python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana
# 单段处理：追加 --episode episodes/pnp_apple_banana/episode_20261101_165909。
import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.rgbd import RGBDFrame
from cordvip_geometry.tracking import SceneTracker, TrackingFailure

from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import RawReader, require_published_raw
from dexmani_real.geometry.hoi import load_task_template, task_objects


def annotate_episodes(episodes, task):
    template = load_task_template(task)
    by_name = {obj["name"]: obj for obj in task_objects(task)}
    objects = [by_name[name] for name in template["object_names"]]
    assets = {obj["name"]: task / obj["name"] for obj in objects}
    prompts = {obj["name"]: obj["prompt"] for obj in objects}
    with ExitStack() as stack:
        tracker = None
        for episode in episodes:
            episode = Path(episode)
            require_published_raw(episode)
            with RawReader(episode) as reader:
                if "hoi" in reader.data_file:
                    reader.require_fields("object_pointcloud")
                    reader["object_pointcloud"]
                    print(f"跳过已发布 HOI：{episode}")
                    continue
                camera = reader.read_camera()
                if camera.T_xarm_base_from_color is None:
                    raise ValueError("HOI 必须有 Raw 中记录的相机外参")
                frames, count = reader.num_frames, len(objects)
                if frames == 0:
                    raise ValueError("空 Episode")
                poses = np.empty((frames, count, 4, 4), np.float32)
                failure = None
                reader.require_fields("rgb", "depth")
                intrinsics = camera.color
                K = intrinsics.matrix()
                expected_shape = (intrinsics.height, intrinsics.width)
                depth_scale = np.float32(camera.depth_scale_m)
                depth = reader["depth"]
                if depth.shape != (frames, *expected_shape) or depth.dtype != np.uint16:
                    raise ValueError("Raw depth must be uint16 [rows,H,W] matching color intrinsics")
                timing = reader["timing/rows"][:]
                images = iter(reader.iter_rgb_frames())
                if tracker is None:
                    tracker = stack.enter_context(
                        SceneTracker.open(assets, workspace=ROOT)
                    )
                else:
                    tracker.reset()
                previous_sequence = previous_stamp = -1
                for index in range(frames):
                    sequence = int(timing[index]["camera_sequence"])
                    stamp = int(timing[index]["camera_ns"])
                    if (
                        sequence <= 0
                        or stamp <= 0
                        or sequence < previous_sequence
                        or stamp < previous_stamp
                    ):
                        raise ValueError(f"非法相机时钟，行 {index}")
                    if sequence == previous_sequence and stamp != previous_stamp:
                        raise ValueError("相同相机序列的来源时间冲突")
                    rgb = next(images, None)
                    if rgb is None:
                        raise ValueError(f"Raw RGB ended before row {index}")
                    if rgb.dtype != np.uint8 or rgb.shape != (*expected_shape, 3):
                        raise ValueError(f"Raw RGB grid invalid at row {index}")
                    if sequence == previous_sequence:
                        poses[index] = poses[index - 1]
                        continue
                    frame = RGBDFrame(
                        rgb=rgb,
                        depth_m=np.multiply(depth[index], depth_scale, dtype=np.float32),
                        K=K,
                        distortion_model=intrinsics.distortion_model,
                        distortion_coeffs=intrinsics.distortion_coeffs,
                    )
                    try:
                        masks = tracker.detect_masks(frame, prompts) if index == 0 else None
                        result = tracker.step(frame, initial_masks=masks)
                    except TrackingFailure as error:
                        failure = str(error)
                        break
                    poses[index] = np.stack([result[name] for name in template["object_names"]])
                    previous_sequence, previous_stamp = sequence, stamp
                else:
                    if next(images, None) is not None:
                        raise ValueError("Raw RGB contains more frames than the HDF5 rows")
            if failure is not None:
                print(f"{episode}: 首次失败行 {index}: {failure}；不发布 HOI")
                continue
            write_hoi(episode / "data.h5", template, poses, camera.T_xarm_base_from_color)
            print(f"{episode}: 已发布 HOI，{frames} 帧")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args(argv)
    if Path(args.task).name != args.task or args.task in (".", ".."):
        parser.error("非法任务名")
    task = ROOT / "assets/pointcloud_template/object" / args.task
    episodes = (
        [args.episode]
        if args.episode
        else sorted((ROOT / "episodes" / args.task).glob("episode_*"))
    )
    if not episodes:
        raise ValueError("没有 Episode")
    annotate_episodes(episodes, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

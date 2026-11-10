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
from cordvip_geometry.raw import iter_raw_rows
from cordvip_geometry.tracking import SceneInitializationError, SceneTracker
from dexmani_real.data.raw import RawReader, require_published_raw
from dexmani_real.data.hoi import write_hoi
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
                if "object_pointcloud" in reader.fields:
                    reader.require_fields("object_pointcloud", "object_pointcloud_valid")
                    print(f"跳过已发布 HOI：{episode}")
                    continue
                camera = reader.read_camera()
                if camera.T_xarm_base_from_color is None:
                    raise ValueError("HOI 必须有 Raw 中记录的相机外参")
                frames, count = reader.num_frames, len(objects)
                if frames == 0:
                    raise ValueError("空 Episode")
                poses = np.full((frames, count, 4, 4), np.nan, np.float32)
                failure = None
                if tracker is None:
                    tracker = stack.enter_context(
                        SceneTracker.open(assets, workspace=ROOT)
                    )
                else:
                    tracker.reset()
                previous_sequence = previous_stamp = -1
                for index, (frame, timing) in enumerate(iter_raw_rows(reader, camera)):
                    sequence, stamp = int(timing["camera_sequence"]), int(timing["camera_ns"])
                    if (
                        sequence <= 0
                        or stamp <= 0
                        or sequence < previous_sequence
                        or stamp < previous_stamp
                    ):
                        raise ValueError(f"非法相机时钟，行 {index}")
                    if sequence == previous_sequence:
                        if stamp != previous_stamp:
                            raise ValueError("相同相机序列的来源时间冲突")
                    else:
                        try:
                            masks = tracker.detect_masks(frame, prompts) if index == 0 else None
                            result = tracker.step(frame, initial_masks=masks)
                        except SceneInitializationError as error:
                            failure = str(error)
                            break
                    failed = [(name, result["objects"][name]["error"]) for name in prompts
                              if not result["objects"][name]["pose_valid"]]
                    if failed:
                        failure = str(failed)
                        break
                    poses[index] = np.stack([result["objects"][name]["pose"] for name in prompts])
                    previous_sequence, previous_stamp = sequence, stamp
            if failure is not None:
                print(f"{episode}: 首次失败行 {index}: {failure}；不发布 HOI")
                continue
            write_hoi(episode / "data.h5", template, poses, camera.T_xarm_base_from_color)
            print(f"{episode}: {frames}/{frames} 帧有效")


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

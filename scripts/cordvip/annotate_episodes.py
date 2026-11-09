"""将已发布 Raw 的离线 HOI 结果写入原 data.h5；不创建 Sidecar。"""

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
from cordvip_geometry.tracking import SceneTracker
from dexmani_real.data.raw import RawReader, require_published_raw
from dexmani_real.data.hoi import write_hoi
from dexmani_real.geometry.hoi import load_task_template, task_objects


def annotate_episodes(episodes, task):
    template, manifest = load_task_template(task)
    objects = task_objects(task)
    assets = {obj["name"]: task / obj["name"] for obj in objects}
    prompts = {obj["name"]: obj.get("prompt", obj["name"]) for obj in objects}
    with ExitStack() as stack:
        tracker = None
        for episode in episodes:
            episode = Path(episode)
            require_published_raw(episode)
            with RawReader(episode) as reader:
                if reader.hoi_metadata is not None:
                    if reader.hoi_metadata != manifest:
                        raise ValueError(f"{episode}: 已有 HOI 的任务或模板不同，拒绝覆盖")
                    print(f"跳过完整 HOI：{episode}")
                    continue
                camera = reader.read_camera()
                if camera.T_xarm_base_from_color is None:
                    raise ValueError("HOI 必须有 Raw 中记录的相机外参")
                if reader.meta["task_label"] != manifest["task_name"]:
                    raise ValueError("Raw task_label 与资产不匹配")
                frames, count = reader.num_frames, len(objects)
                if frames == 0:
                    raise ValueError("空 Episode")
                poses = np.full((frames, count, 4, 4), np.nan, np.float32)
                valid = np.zeros((frames, count), bool)
                quality = np.zeros((frames, count), np.float32)
                last_errors = dict.fromkeys(prompts)
                if tracker is None:
                    tracker = stack.enter_context(
                        SceneTracker.open(assets, workspace=ROOT, max_reregistrations=3)
                    )
                else:
                    tracker.reset()
                previous_sequence = previous_stamp = -1
                for index, (frame, timing) in enumerate(iter_raw_rows(reader, camera)):
                    sequence, stamp = int(timing["camera_sequence"]), int(timing["camera_ns"])
                    if (
                        sequence < 0
                        or stamp < 0
                        or sequence < previous_sequence
                        or stamp < previous_stamp
                    ):
                        raise ValueError(f"非法相机时钟，行 {index}")
                    if sequence == previous_sequence:
                        if stamp != previous_stamp:
                            raise ValueError("相同相机序列的来源时间冲突")
                    else:
                        masks = tracker.detect_masks(frame, prompts) if index == 0 else None
                        result = tracker.step(frame, initial_masks=masks)
                    for j, name in enumerate(prompts):
                        item = result["objects"][name]
                        if item["error"] is not None:
                            last_errors[name] = item["error"]
                        valid[index, j] = item["pose_valid"]
                        quality[index, j] = item["cloud"].model_supported_fraction
                        if item["pose_valid"]:
                            poses[index, j] = item["pose"]
                    previous_sequence, previous_stamp = sequence, stamp
                timing_rows = reader["timing/rows"].fields(["camera_ns", "arm_ns", "hand_ns"])[:]
            write_hoi(
                episode / "data.h5",
                template,
                manifest,
                poses,
                valid,
                quality,
                camera.T_xarm_base_from_color,
            )
            print(f"{episode}: {valid.all(axis=1).sum()}/{frames} 帧有效")
            for j, name in enumerate(prompts):
                lost_rows = np.flatnonzero(~valid[:, j])
                # 每个任务物体均加载 Mesh，pose_valid 与 TRACKING 等价。
                tracking_rows = frames - len(lost_rows)
                first_lost = int(lost_rows[0]) if len(lost_rows) else None
                state = result["objects"][name]["state"]
                attempts = tracker.registration_attempts.get(name, 0)
                exhausted = state == "POSE_LOST" and attempts >= tracker.max_reregistrations
                print(
                    f"  {name}: 有效 {tracking_rows}/{frames}; "
                    f"首次丢失 Raw 行(0-based)={first_lost}; "
                    f"TRACKING={tracking_rows}, POSE_LOST={len(lost_rows)}; "
                    f"末态={state}; 最近失败原因={last_errors[name]}; "
                    f"当前连续重注册={attempts}/{tracker.max_reregistrations}, 耗尽={exhausted}"
                )
            for field in ("arm_ns", "hand_ns"):
                available = (timing_rows["camera_ns"] >= 0) & (timing_rows[field] >= 0)
                delta_ms = np.abs(timing_rows["camera_ns"][available] - timing_rows[field][available]) / 1e6
                summary = (
                    "/".join(f"{value:.3f}" for value in np.percentile(delta_ms, [50, 95, 100]))
                    if len(delta_ms) else "不可用"
                )
                print(
                    f"  主机记录时间戳差 |camera_ns-{field}|: "
                    f"P50/P95/Max(ms)={summary}, 有效样本={len(delta_ms)}/{frames}（非硬件同步误差）"
                )


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

"""从现有 Camera Ring 生成带来源时钟及 Episode 标识的 HOI。"""

import signal
import sys
import time
from pathlib import Path

import numpy as np


def run_hoi_worker(shared, camera, task, template):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "third_party/cordvip_geometry/src"))
    from cordvip_geometry.rgbd import RGBDFrame
    from cordvip_geometry.tracking import SceneTracker
    from dexmani_real.geometry.hoi import task_objects, transform_task_points

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    objects = task_objects(task)
    prompts = {obj["name"]: obj.get("prompt", obj["name"]) for obj in objects}
    camera_fields = tuple(name for name in ("rgb", "depth") if name in shared.hoi_ring.dtype.names)
    epoch = last_sequence = 0
    first = True
    with SceneTracker.open({name: task / name for name in prompts}, workspace=root) as tracker:
        shared.hoi_ready.value = True
        while shared.is_running.value:
            request = shared.hoi_request.read()
            if request is None:
                time.sleep(0.005)
                continue
            command = request[0]
            requested_epoch = int(command["episode_id"])
            if epoch != requested_epoch:
                tracker.reset()
                epoch = requested_epoch
                first = True
                last_sequence = int(command["after_sequence"])
            result = shared.camera_ring.read()
            if result is None or result[1] <= last_sequence:
                time.sleep(0.001)
                continue
            data, sequence = result
            last_sequence = sequence
            if int(data["timestamp_ns"]) <= int(command["after_ns"]):
                continue
            frame = RGBDFrame.from_camera(camera, {name: data[name] for name in data.dtype.names})
            masks = tracker.detect_masks(frame, prompts) if first else None
            scene = tracker.step(frame, initial_masks=masks)
            first = False
            valid = all(scene["objects"][name]["pose_valid"] for name in prompts)
            points = np.zeros((1024, 3), np.float32)
            if valid:
                poses = np.stack([scene["objects"][name]["pose"] for name in prompts])
                points = transform_task_points(template, poses, camera.T_xarm_base_from_color)
            # 不将上一轮在途结果发布为新轮次观测。
            if int(shared.hoi_request.read()[0]["episode_id"]) != epoch:
                continue
            shared.hoi_ring.write(
                episode_id=epoch,
                timestamp_ns=data["timestamp_ns"],
                source_camera_sequence=sequence,
                valid=valid,
                object_pointcloud=points,
                **{name: data[name] for name in camera_fields},
            )

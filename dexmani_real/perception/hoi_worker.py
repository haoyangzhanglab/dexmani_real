"""从 Camera Ring 生成当前 Episode 的物体点云。"""

import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_hoi_worker(shared, camera, task, template):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "third_party/cordvip_geometry/src"))
    from cordvip_geometry.camera import RGBDFrame
    from cordvip_geometry.tracking import SceneTracker, TrackingFailure

    from dexmani_real.geometry.hoi import task_objects, transform_task_points

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    by_name = {obj["name"]: obj for obj in task_objects(task)}
    prompts = {name: by_name[name]["prompt"] for name in template["object_names"]}
    camera_fields = tuple(
        name for name in ("rgb", "depth") if name in shared.hoi_ring.field_names
    )
    epoch = last_sequence = 0
    first = True
    episode_failed = False
    tracker = SceneTracker(
        {name: task / name / "mesh/mesh_m.obj" for name in prompts}, workspace=root
    )
    shared.hoi_ready.value = True
    while shared.is_running.value:
        request = shared.hoi_request.read()
        if request is None or int(request[0]["episode_id"]) == 0:
            time.sleep(0.005)
            continue
        command = request[0]
        requested_epoch = int(command["episode_id"])
        if epoch != requested_epoch:
            epoch = requested_epoch
            first = True
            episode_failed = False
            last_sequence = int(command["after_sequence"])
        if episode_failed:
            time.sleep(0.005)
            continue
        result = shared.camera_ring.read()
        if result is None or result[1] <= last_sequence:
            time.sleep(0.001)
            continue
        data, sequence = result
        last_sequence = sequence
        if int(data["timestamp_ns"]) <= int(command["after_ns"]):
            continue
        frame = RGBDFrame.from_camera(camera, data)
        failure = None
        try:
            scene = (
                tracker.initialize(frame, prompts) if first else tracker.track(frame)
            )
        except TrackingFailure as error:
            failure = str(error)
            points = np.zeros((1024, 3), np.float32)
        else:
            poses = np.stack([scene.poses[name] for name in template["object_names"]])
            points = transform_task_points(
                template, poses, camera.T_xarm_base_from_color
            )
        first = False
        # 丢弃已经切换 Episode 的在途结果。
        if int(shared.hoi_request.read()[0]["episode_id"]) != epoch:
            continue
        if failure is not None:
            episode_failed = True
            logger.warning(
                "HOI Episode %d 失败：%s；等待下一轮 B 请求。", epoch, failure
            )
        shared.hoi_ring.write(
            episode_id=epoch,
            timestamp_ns=data["timestamp_ns"],
            source_camera_sequence=sequence,
            valid=failure is None,
            object_pointcloud=points,
            **{name: data[name] for name in camera_fields},
        )

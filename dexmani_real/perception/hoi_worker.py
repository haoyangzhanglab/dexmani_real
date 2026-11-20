"""从 Camera Ring 生成物体点云，每个新请求独立初始化跟踪。"""

from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.perception.hoi import (
    build_tracking_config,
    invalid_hoi_frame,
    track_hoi_frame,
)

if TYPE_CHECKING:
    from dexmani_real.config.perception import HOITrackingConfig
    from dexmani_real.geometry.camera import CameraSnapshot

logger = logging.getLogger(__name__)


def run_hoi_worker(
    shared,
    camera: CameraSnapshot,
    template: Mapping[str, np.ndarray],
    mesh_paths: dict[str, Path],
    prompts: dict[str, str],
    tracking_config: HOITrackingConfig,
) -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "third_party/cordvip_geometry/src"))
    from cordvip_geometry.camera import RGBDFrame
    from cordvip_geometry.tracking import SceneTracker, TrackingFailure

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    publish_rgb = "rgb" in shared.hoi_ring.field_names
    active_request = last_sequence = 0
    initialize = True
    tracker = SceneTracker(
        mesh_paths, workspace=root, config=build_tracking_config(tracking_config)
    )
    invalid_points = np.zeros((HOI_NUM_POINTS, 3), np.float32)
    shared.hoi_ready.value = True
    while shared.is_running.value:
        request = shared.hoi_request.read()
        if request is None or not request[0]["episode_id"]:
            time.sleep(0.005)
            continue
        command, request_sequence = request
        if request_sequence != active_request:
            active_request = request_sequence
            initialize = True
            last_sequence = int(command["after_sequence"])
        camera_result = shared.camera_ring.read()
        if camera_result is None or camera_result[1] <= last_sequence:
            time.sleep(0.001)
            continue
        data, sequence = camera_result
        last_sequence = sequence
        if int(data["timestamp_ns"]) <= int(command["after_ns"]):
            continue
        frame = RGBDFrame.from_camera(camera, data, source_sequence=sequence)
        try:
            result = track_hoi_frame(tracker, frame, template, prompts, initialize=initialize)
        except TrackingFailure as error:
            result = invalid_hoi_frame(template, str(error))
        else:
            initialize = False
        valid = result.valid
        points = (
            transform_task_points(
                template, result.object_pose_camera, camera.T_xarm_base_from_color
            )
            if valid
            else invalid_points
        )
        # 将请求序列检查和结果发布放在同一临界区。
        with shared.hoi_request_lock:
            if shared.hoi_request.latest_sequence != request_sequence:
                continue
            shared.hoi_ring.write(
                episode_id=command["episode_id"],
                timestamp_ns=data["timestamp_ns"],
                source_camera_sequence=sequence,
                valid=valid,
                object_pointcloud=points,
                **({"rgb": data["rgb"]} if publish_rgb else {}),
            )
        if not valid:
            logger.warning(
                "HOI Episode %d 当前帧无效：%s；继续处理后续帧。",
                command["episode_id"], result.failures,
            )

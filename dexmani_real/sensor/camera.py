import json

from dexmani_real.sensor.config import CameraConfig
from dexmani_real.utils.log import configure_logging


def run_camera_worker(shared, config: CameraConfig):
    from dexmani_real.sensor.realsense import RealSenseCamera

    configure_logging()
    camera = RealSenseCamera(config)
    camera.connect()
    try:
        shared.camera_depth_scale.value = camera.depth_scale_m
        shared.camera_serial.value = camera.active_serial.encode()
        shared.camera_intrinsics.value = json.dumps(camera.color_intrinsics.to_dict()).encode()
        while shared.is_running.value:
            frame = camera.read()
            if frame is None:
                continue
            shared.camera_ring.write(
                timestamp_ns=frame.timestamp_ns,
                color_frame_number=frame.color_frame_number,
                rgb=frame.rgb,
                depth=frame.depth,
            )
            shared.camera_ready.set()
    finally:
        shared.camera_ready.clear()
        camera.disconnect()

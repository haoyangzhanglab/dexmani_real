"""RealSense owner; timestamp each channel when its frame first reaches the host."""

import logging
import json
import time

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.utils.log import configure_logging


logger = logging.getLogger(__name__)


def run_camera_worker(shared, config: CameraParams) -> None:
    configure_logging()
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

    cam = RealSenseCamera(config)
    cam.connect()
    try:
        shared.camera_depth_scale.value = cam.get_depth_scale()
        shared.camera_serial.value = cam.active_serial.encode()
        shared.camera_geometry.value = json.dumps(cam.get_geometry().to_dict()).encode()
        last_frame = None
        last_advance_ns = [time.monotonic_ns(), time.monotonic_ns()]
        while shared.is_running.value:
            try:
                frame = cam.read(timeout_ms=300)
            except (RuntimeError, OSError):
                if time.monotonic_ns() - min(last_advance_ns) >= int(
                    config.source_stall_timeout_s * 1e9
                ):
                    raise
                time.sleep(0.01)
                continue
            identity = (frame.depth_frame_number, frame.color_frame_number)
            now = time.monotonic_ns()
            for channel, name in enumerate(("depth", "RGB")):
                if identity[channel] < 0:
                    raise RuntimeError(f"RealSense missing {name} frame identity")
                if last_frame is not None and identity[channel] < last_frame[channel]:
                    raise RuntimeError(f"RealSense {name} frame identity moved backward")
                if last_frame is None or identity[channel] != last_frame[channel]:
                    last_advance_ns[channel] = int(frame.timestamp_ns)
                elif now - last_advance_ns[channel] >= int(config.source_stall_timeout_s * 1e9):
                    raise RuntimeError(f"RealSense stopped producing new {name}")
            if identity == last_frame:
                continue
            header = np.array(
                (min(last_advance_ns), frame.timestamp_ns, *identity),
                dtype=CAMERA_FRAME_HEADER_DTYPE,
            )
            shared.camera_ring.write(
                header=header, rgb=frame.rgb, depth=frame.depth_aligned_to_color_raw,
            )
            last_frame = identity
            shared.camera_ready.set()
    except Exception:
        logger.exception("camera worker failed")
        raise
    finally:
        shared.camera_ready.clear()
        cam.disconnect()

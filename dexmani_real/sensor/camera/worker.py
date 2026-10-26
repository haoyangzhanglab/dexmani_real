"""RealSense owner; timestamp each channel when its frame first reaches the host."""

import json
import logging
import time
from typing import cast

from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


def run_camera_worker(shared, config: CameraParams) -> None:
    """持有相机流，按各通道首次接收时刻发布观测。"""
    configure_logging()
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

    camera = RealSenseCamera(config)
    camera.connect()
    try:
        shared.camera_depth_scale.value = camera.get_depth_scale()
        shared.camera_serial.value = cast(str, camera.active_serial).encode()
        shared.camera_intrinsics.value = json.dumps(
            camera.get_color_intrinsics().to_dict()
        ).encode()
        last_frame = None
        last_advance_ns = [time.monotonic_ns(), time.monotonic_ns()]
        while shared.is_running.value:
            try:
                frame = camera.read(timeout_ms=300)
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
                elif now - last_advance_ns[channel] >= int(
                    config.source_stall_timeout_s * 1e9
                ):
                    raise RuntimeError(f"RealSense stopped producing new {name}")
            if identity == last_frame:
                continue
            shared.camera_ring.write(
                timestamp_ns=min(last_advance_ns),
                color_frame_number=frame.color_frame_number,
                rgb=frame.rgb,
                depth=frame.depth_aligned_to_color_raw,
            )
            last_frame = identity
            shared.camera_ready.set()
    except Exception:
        logger.exception("camera worker failed")
        raise
    finally:
        shared.camera_ready.clear()
        camera.disconnect()

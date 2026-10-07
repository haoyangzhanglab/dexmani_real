"""RealSense owner; publish aligned RGB-D with the oldest channel advance time."""

import json
import time

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def pack_camera_frame(rgb, depth_raw, *, timestamp_ns, depth_frame_number, color_frame_number):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("camera RGB must be uint8 [H,W,3]")
    if depth_raw.dtype != np.uint16 or depth_raw.shape != rgb.shape[:2]:
        raise ValueError("aligned depth must be uint16 [H,W]")
    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["timestamp_ns"] = timestamp_ns
    header["depth_frame_number"] = depth_frame_number
    header["color_frame_number"] = color_frame_number
    return header, rgb, depth_raw


def run_camera_worker(shared, config: CameraParams) -> None:
    if shared.camera_ring is None:
        raise ValueError("camera worker requires an allocated camera ring")
    from dexmani_real.sensor.camera.realsense import (
        RealSenseCamera,
        RealSenseCameraConfig,
    )

    cam = RealSenseCamera(RealSenseCameraConfig.from_camera_params(config))
    try:
        if not cam.connect():
            raise RuntimeError("RealSense connect failed")
        shared.camera_depth_scale.value = cam.get_depth_scale()
        shared.camera_serial.value = str(cam.active_serial or "").encode()
        shared.camera_geometry.value = json.dumps(cam.get_geometry().to_dict()).encode()
        last_frame = None
        last_advance_ns = [time.monotonic_ns(), time.monotonic_ns()]
        while shared.is_running.value:
            try:
                frame = cam.read(timeout_ms=300, compute_depth=False)
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
                if identity[channel] is None or identity[channel] < 0:
                    raise RuntimeError(f"RealSense missing {name} frame identity")
                if last_frame is None or identity[channel] != last_frame[channel]:
                    last_advance_ns[channel] = int(frame.timestamp_ns)
                elif now - last_advance_ns[channel] >= int(config.source_stall_timeout_s * 1e9):
                    raise RuntimeError(f"RealSense stopped producing new {name}")
            if identity == last_frame:
                continue
            if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
                raise RuntimeError("RealSense did not produce aligned RGB-D")
            shared.camera_ring.write(
                *pack_camera_frame(
                    frame.rgb,
                    frame.depth_aligned_to_color_raw,
                    timestamp_ns=min(last_advance_ns),
                    depth_frame_number=frame.depth_frame_number,
                    color_frame_number=frame.color_frame_number,
                )
            )
            last_frame = identity
            shared.camera_ready.set()
    except Exception:
        logger.exception("camera worker failed")
        raise
    finally:
        shared.camera_ready.clear()
        cam.disconnect()

"""RealSense owner; publish aligned RGB-D with its oldest channel source time."""

import logging
import json
import time

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.utils.log import configure_logging


logger = logging.getLogger(__name__)


def pack_camera_frame(frame, *, timestamp_ns):
    rgb, depth_raw = frame.rgb, frame.depth_aligned_to_color_raw
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("camera RGB must be uint8 [H,W,3]")
    if depth_raw.dtype != np.uint16 or depth_raw.shape != rgb.shape[:2]:
        raise ValueError("aligned depth must be uint16 [H,W]")
    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["timestamp_ns"] = timestamp_ns
    header["received_ns"] = frame.timestamp_ns
    header["depth_frame_number"] = frame.depth_frame_number
    header["color_frame_number"] = frame.color_frame_number
    header["depth_sdk_ns"] = round(frame.depth_device_timestamp_s * 1e9)
    header["color_sdk_ns"] = round(frame.color_device_timestamp_s * 1e9)
    header["depth_timestamp_domain"] = frame.depth_timestamp_domain
    header["color_timestamp_domain"] = frame.color_timestamp_domain
    return header, rgb, depth_raw


def run_camera_worker(shared, config: CameraParams) -> None:
    configure_logging()
    if shared.camera_ring is None:
        raise ValueError("camera worker requires an allocated camera ring")
    from dexmani_real.sensor.camera.realsense import (
        RealSenseCamera,
    )

    cam = RealSenseCamera(config)
    try:
        if not cam.connect():
            raise RuntimeError("RealSense connect failed")
        shared.camera_depth_scale.value = cam.get_depth_scale()
        shared.camera_serial.value = str(cam.active_serial or "").encode()
        shared.camera_geometry.value = json.dumps(cam.get_geometry().to_dict()).encode()
        last_frame = None
        last_advance_ns = [time.monotonic_ns(), time.monotonic_ns()]
        source_ns = [0, 0]
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
            frame_sources = (frame.depth_source_ns, frame.color_source_ns)
            now = time.monotonic_ns()
            for channel, name in enumerate(("depth", "RGB")):
                if identity[channel] is None or identity[channel] < 0:
                    raise RuntimeError(f"RealSense missing {name} frame identity")
                if last_frame is not None and identity[channel] < last_frame[channel]:
                    raise RuntimeError(f"RealSense {name} frame identity moved backward")
                if last_frame is None or identity[channel] != last_frame[channel]:
                    if not 0 < frame_sources[channel] <= frame.timestamp_ns:
                        raise RuntimeError(f"RealSense invalid or future {name} source timestamp")
                    last_advance_ns[channel] = int(frame.timestamp_ns)
                    source_ns[channel] = int(frame_sources[channel])
                elif now - last_advance_ns[channel] >= int(config.source_stall_timeout_s * 1e9):
                    raise RuntimeError(f"RealSense stopped producing new {name}")
            if identity == last_frame:
                continue
            if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
                raise RuntimeError("RealSense did not produce aligned RGB-D")
            header, rgb, depth = pack_camera_frame(
                frame, timestamp_ns=min(source_ns),
            )
            shared.camera_ring.write_fields(header=header[0], rgb=rgb, depth=depth)
            last_frame = identity
            shared.camera_ready.set()
    except Exception:
        logger.exception("camera worker failed")
        raise
    finally:
        shared.camera_ready.clear()
        cam.disconnect()

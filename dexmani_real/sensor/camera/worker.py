"""RealSense owner; publish usable aligned RGB-D and host acquisition times."""

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
    header["rgb_size"], header["depth_size"] = rgb.nbytes, depth_raw.nbytes
    header["rgb_shape_h"], header["rgb_shape_w"], header["rgb_shape_c"] = rgb.shape
    header["depth_shape_h"], header["depth_shape_w"] = depth_raw.shape
    return header, rgb, depth_raw


def run_camera_worker(shared, config: CameraParams) -> None:
    config.validate()
    from dexmani_real.sensor.camera.realsense import (
        L515DepthConfig,
        RealSenseCamera,
        RealSenseCameraConfig,
    )

    cfg = config
    cam = RealSenseCamera(
        RealSenseCameraConfig(
            camera_name="realsense",
            serial=cfg.serial,
            depth_resolution=(cfg.width, cfg.height),
            color_resolution=(cfg.width, cfg.height),
            fps=cfg.fps,
            warmup_frames=cfg.warmup_frames,
            frame_queue_capacity=cfg.frame_queue_capacity,
            l515_depth_config=L515DepthConfig(
                visual_preset=cfg.l515_visual_preset,
                confidence_threshold=cfg.l515_confidence_threshold,
            ),
        )
    )
    try:
        if not cam.connect():
            raise RuntimeError("RealSense connect failed")
        shared.camera_depth_scale.value = cam.get_depth_scale()
        shared.camera_serial.value = str(cam.active_serial or "").encode()
        shared.camera_geometry.value = json.dumps(cam.get_geometry().to_dict()).encode()
        last_frame = None
        last_good_s = time.monotonic()
        while shared.is_running.value:
            try:
                frame = cam.read(timeout_ms=300, compute_depth=False)
            except (RuntimeError, OSError):
                if time.monotonic() - last_good_s >= cfg.source_stall_timeout_s:
                    raise
                time.sleep(0.01)
                continue
            identity = (frame.depth_frame_number, frame.color_frame_number)
            if identity == last_frame:
                if time.monotonic() - last_good_s >= cfg.source_stall_timeout_s:
                    raise RuntimeError("RealSense stopped producing new RGB-D")
                continue
            if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
                raise RuntimeError("RealSense did not produce aligned RGB-D")
            shared.camera_ring.write(
                *pack_camera_frame(
                    frame.rgb,
                    frame.depth_aligned_to_color_raw,
                    timestamp_ns=frame.timestamp_ns,
                    depth_frame_number=frame.depth_frame_number,
                    color_frame_number=frame.color_frame_number or 0,
                )
            )
            last_frame, last_good_s = identity, time.monotonic()
            shared.camera_ready.set()
    except Exception:
        logger.exception("camera worker failed")
        raise
    finally:
        shared.camera_ready.clear()
        cam.disconnect()

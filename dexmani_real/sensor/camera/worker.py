"""RealSense owner; timestamp each channel when its frame first reaches the host."""

import logging
import json
import time

import numpy as np

from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


class CameraWorker:
    """Own one RealSense stream and publish host receive times per channel."""

    def __init__(self, shared, config: CameraParams):
        self.shared = shared
        self.config = config

    def run(self):
        configure_logging()
        from dexmani_real.sensor.camera.realsense import RealSenseCamera

        self.camera = RealSenseCamera(self.config)
        self.camera.connect()
        try:
            self.shared.camera_depth_scale.value = self.camera.get_depth_scale()
            self.shared.camera_serial.value = self.camera.active_serial.encode()
            self.shared.camera_intrinsics.value = json.dumps(
                self.camera.get_geometry().color.to_dict()
            ).encode()
            self.last_frame = None
            self.last_advance_ns = [time.monotonic_ns(), time.monotonic_ns()]
            while self.shared.is_running.value:
                try:
                    frame = self.camera.read(timeout_ms=300)
                except (RuntimeError, OSError):
                    if time.monotonic_ns() - min(self.last_advance_ns) >= int(
                        self.config.source_stall_timeout_s * 1e9
                    ):
                        raise
                    time.sleep(0.01)
                    continue
                identity = (frame.depth_frame_number, frame.color_frame_number)
                now = time.monotonic_ns()
                for channel, name in enumerate(("depth", "RGB")):
                    if identity[channel] < 0:
                        raise RuntimeError(f"RealSense missing {name} frame identity")
                    if self.last_frame is not None and identity[channel] < self.last_frame[channel]:
                        raise RuntimeError(f"RealSense {name} frame identity moved backward")
                    if self.last_frame is None or identity[channel] != self.last_frame[channel]:
                        self.last_advance_ns[channel] = int(frame.timestamp_ns)
                    elif now - self.last_advance_ns[channel] >= int(
                        self.config.source_stall_timeout_s * 1e9
                    ):
                        raise RuntimeError(f"RealSense stopped producing new {name}")
                if identity == self.last_frame:
                    continue
                header = np.array(
                    (min(self.last_advance_ns), frame.timestamp_ns, *identity),
                    dtype=CAMERA_FRAME_HEADER_DTYPE,
                )
                self.shared.camera_ring.write(
                    header=header,
                    rgb=frame.rgb,
                    depth=frame.depth_aligned_to_color_raw,
                )
                self.last_frame = identity
                self.shared.camera_ready.set()
        except Exception:
            logger.exception("camera worker failed")
            raise
        finally:
            self.shared.camera_ready.clear()
            self.camera.disconnect()


def run_camera_worker(shared, config: CameraParams) -> None:
    CameraWorker(shared, config).run()

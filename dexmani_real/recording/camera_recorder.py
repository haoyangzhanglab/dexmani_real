"""CameraRecorder — background camera acquisition via SharedRingBuffer.

Runs RealSense capture in a subprocess, writing frames to a ring buffer
for the controller process to consume during recording.
"""

from __future__ import annotations

import multiprocessing
import pickle
import sys
import time
from typing import Any

import numpy as np


class CameraRecorder:
    """Background camera capture process.

    Usage:
        recorder = CameraRecorder(camera_config, ring_name="camera")
        recorder.start()
        # ... teleop loop reads frames via ring buffer ...
        recorder.stop()
    """

    def __init__(
        self,
        camera_config: Any | None = None,
        ring_name: str = "camera",
        target_fps: float = 30.0,
    ) -> None:
        self.camera_config = camera_config
        self.ring_name = ring_name
        self.target_fps = target_fps

        self._process: multiprocessing.Process | None = None
        self._running = multiprocessing.Event()
        self._ready = multiprocessing.Event()

    def start(self) -> None:
        if self._process is not None:
            return
        self._running.set()
        self._ready.clear()
        self._process = multiprocessing.Process(
            target=self._capture_loop,
            name="camera-recorder",
            daemon=True,
        )
        self._process.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        self._running.clear()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _capture_loop(self) -> None:
        from dexmani_real.ipc.shared_ring_buffer import RingBufferConfig, SharedRingBuffer
        from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

        config = self.camera_config or RealSenseConfig()
        camera = RealSense(config)
        if not camera.connect():
            return

        ring_config = RingBufferConfig(slot_count=256, slot_size=4_194_304, create=True)
        ring = SharedRingBuffer(self.ring_name, ring_config)

        self._ready.set()

        dt = 1.0 / self.target_fps
        last_time = time.perf_counter()

        try:
            while self._running.is_set():
                frame = camera.read()
                data = {
                    "rgb": frame.rgb,
                    "depth": frame.depth,
                    "timestamp": frame.timestamp,
                    "K": frame.K.copy() if frame.K is not None else None,
                }
                ring.write(pickle.dumps(data))

                elapsed = time.perf_counter() - last_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                last_time = time.perf_counter()
        except Exception as e:
            print(f"[CameraRecorder] capture error: {e}", file=sys.stderr)
        finally:
            camera.disconnect()
            ring.close()

    @staticmethod
    def read_frame(ring_buffer: Any) -> dict[str, Any] | None:
        """Read latest camera frame from ring buffer. Called from controller."""
        data, _ = ring_buffer.read(last_seq=-1)
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception:
            return None

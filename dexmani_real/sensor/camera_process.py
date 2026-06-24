"""Camera recording daemon process — crash-isolated frame capture.

Runs RealSense capture in a separate multiprocessing.Process so that USB
disconnects, firmware hangs, or frame timeouts don't crash the control loop.

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem).

Architecture (Queue mode, default):
    ┌───────────────────────┐  mp.Queue(maxsize=1)  ┌──────────────────────┐
    │ CameraProcess         │──────────────────────►│ TeleopController     │
    │ (独立 mp.Process)     │  dict (latest frame)  │ (主进程, 50Hz)       │
    │                       │                       │                      │
    │ RealSense.read()      │                       │ poll_latest_frame()  │
    │ → CameraFrame.to_dict│                       │ → recorder.add_frame│
    └───────────────────────┘                       └──────────────────────┘

Architecture (Shared Memory mode, use_shm=True):
    ┌───────────────────────┐   SharedMemory         ┌──────────────────────┐
    │ CameraProcess         │ ── CameraRingBuffer ──►│ TeleopController     │
    │ (独立 mp.Process)     │   (zero-copy)          │ (主进程, 50Hz)       │
    │                       │                        │                      │
    │ RealSense.read()      │                        │ poll_latest_frame()  │
    │ → shm.write()         │                        │ → shm.read_latest()  │
    └───────────────────────┘                        └──────────────────────┘
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)


@dataclass
class CameraProcessConfig:
    """Configuration for CameraProcess."""

    camera_name: str = "realsense"
    serial: str | None = None
    hz: float = 30.0
    warmup_frames: int = 10
    timeout_ms: int = 1000
    # Shared memory: use CameraRingBuffer (zero-copy) instead of mp.Queue
    use_shm: bool = False
    shm_name: str = "dexmani_cam_0"  # matches SharedMemoryFrameManager defaults
    rgb_height: int = 480
    rgb_width: int = 640


class CameraProcess:
    """Captures RealSense frames in a crash-isolated background process.

    Usage (Queue mode):
        cam = CameraProcess(CameraProcessConfig(serial="...", hz=30.0))
        cam.start()
        # In controller loop:
        frame = cam.poll_latest_frame()
        if frame is not None:
            recorder.add_frame(..., camera_frame=frame)
        cam.stop()

    Usage (Shared Memory mode):
        cam = CameraProcess(CameraProcessConfig(serial="...", hz=30.0, use_shm=True))
        cam.start()
        frame = cam.poll_latest_frame()  # reads from shm
        cam.stop()
    """

    def __init__(self, config: CameraProcessConfig | None = None) -> None:
        self.config = config or CameraProcessConfig()
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()

        # maxsize=1 ensures only the latest frame is queued (old frames
        # dropped automatically when the queue is full).
        self._frame_queue: mp.Queue | None = None
        if not self.config.use_shm:
            self._frame_queue = mp.Queue(maxsize=1)

        # Shared memory ring buffer (zero-copy path)
        self._shm_buf = None  # CameraRingBuffer instance (lazy init)

        # Crash detection
        self._crashed = mp.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the camera process. Returns True on success."""
        if self._process is not None and self._process.is_alive():
            logger.warning("CameraProcess already running.")
            return False

        self._stop_event.clear()
        self._crashed.clear()

        # Drain any stale frames from previous run
        if self._frame_queue is not None:
            self._drain_queue()

        # Initialize shared memory buffer if using shm mode
        if self.config.use_shm:
            self._init_shm()

        self._process = mp.Process(
            target=self._run,
            name=f"camera-{self.config.camera_name}",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "CameraProcess started (name=%s, serial=%s, hz=%.0f, shm=%s)",
            self.config.camera_name, self.config.serial or "default",
            self.config.hz, self.config.use_shm,
        )
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop and wait for process exit."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning(
                    "CameraProcess did not exit within %.1fs, terminating.", timeout
                )
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None
        if self._frame_queue is not None:
            self._drain_queue()
        logger.info("CameraProcess stopped.")

    # ------------------------------------------------------------------
    # Frame access (called from main process)
    # ------------------------------------------------------------------

    def poll_latest_frame(self) -> dict | None:
        """Non-blocking poll for the latest camera frame.

        Returns a dict (CameraFrame.to_dict() format) or None.
        In queue mode: drains the queue to always return the newest frame.
        In shm mode: reads the latest frame from the shared memory ring buffer.
        """
        if self.config.use_shm:
            return self._poll_shm()

        # Queue mode
        latest: dict | None = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    @property
    def crashed(self) -> bool:
        """Whether the camera process has crashed."""
        if self._process is not None and not self._process.is_alive():
            self._crashed.set()
        return self._crashed.is_set()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ------------------------------------------------------------------
    # Shared memory helpers
    # ------------------------------------------------------------------

    def _init_shm(self) -> None:
        """Initialize or attach to the shared memory camera ring buffer."""
        from dexmani_real.shm.ring_buffer import CameraRingBuffer

        h = self.config.rgb_height
        w = self.config.rgb_width
        try:
            self._shm_buf = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=True,
            )
        except FileExistsError:
            self._shm_buf = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=False,
            )

    def _poll_shm(self) -> dict | None:
        """Read latest frame from shared memory."""
        if self._shm_buf is None:
            self._init_shm()
        if self._shm_buf is None:
            return None
        from dexmani_real.shm.layouts import bytes_to_camera_frame

        result = self._shm_buf.read_latest()
        if result is None:
            return None
        header, rgb, depth, seq = result
        return bytes_to_camera_frame(header, rgb, depth)

    # ------------------------------------------------------------------
    # Internal (runs in child process)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main capture loop (runs in child process)."""
        interval = 1.0 / self.config.hz

        try:
            from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

            rs_config = RealSenseConfig(
                camera_name=self.config.camera_name,
                serial=self.config.serial,
                fps=int(self.config.hz),
                warmup_frames=self.config.warmup_frames,
            )

            cam = RealSense(rs_config)
            if not cam.connect():
                logger.error("CameraProcess: RealSense connect failed.")
                self._crashed.set()
                return

            logger.info(
                "CameraProcess capture loop started @ %.0f Hz (shm=%s).",
                self.config.hz, self.config.use_shm,
            )

            # Setup shm writer if using shared memory mode
            shm_writer = None
            if self.config.use_shm:
                from dexmani_real.shm.layouts import camera_frame_to_bytes
                from dexmani_real.shm.ring_buffer import CameraRingBuffer

                h = self.config.rgb_height
                w = self.config.rgb_width
                # Child process: parent already created the shared memory.
                # Using create=False avoids the unnecessary FileExistsError
                # fallback path and eliminates a race window.
                shm_writer = CameraRingBuffer(
                    name=self.config.shm_name,
                    rgb_shape=(h, w, 3),
                    depth_shape=(h, w),
                    maxlen=5,
                    create=False,
                )

            last_ts = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    frame = cam.read(timeout_ms=self.config.timeout_ms)
                    frame_dict = frame.to_dict()

                    if shm_writer is not None:
                        # Shared memory path: zero-copy write
                        try:
                            header, rgb, depth = camera_frame_to_bytes(frame_dict)
                            shm_writer.write(header, rgb, depth)
                        except (ValueError, RuntimeError, OSError):
                            logger.exception(
                                "CameraProcess shm write failed — continuing."
                            )
                    else:
                        # Queue path: non-blocking put, drops oldest when full
                        try:
                            self._frame_queue.put_nowait(frame_dict)
                        except queue.Full:
                            try:
                                self._frame_queue.get_nowait()  # drop oldest
                            except queue.Empty:
                                pass
                            self._frame_queue.put_nowait(frame_dict)
                except (RuntimeError, OSError):
                    logger.exception("CameraProcess frame read failed — continuing.")

                # Maintain target rate
                elapsed = time.monotonic() - last_ts
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_ts = time.monotonic()

            if shm_writer is not None:
                shm_writer.close()
            cam.disconnect()
            logger.info("CameraProcess capture loop exited cleanly.")

        except (RuntimeError, OSError):
            logger.exception("CameraProcess crashed.")
            self._crashed.set()

    def _drain_queue(self) -> None:
        """Remove all pending frames from the queue."""
        if self._frame_queue is None:
            return
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

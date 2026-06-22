"""Camera recording daemon process — crash-isolated frame capture.

Runs RealSense capture in a separate multiprocessing.Process so that USB
disconnects, firmware hangs, or frame timeouts don't crash the control loop.

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem).

Architecture:
    ┌───────────────────────┐  mp.Queue(maxsize=1)  ┌──────────────────────┐
    │ CameraProcess         │──────────────────────►│ TeleopController     │
    │ (独立 mp.Process)     │  dict (latest frame)  │ (主进程, 50Hz)       │
    │                       │                       │                      │
    │ RealSense.read()      │                       │ poll_latest_frame()  │
    │ → CameraFrame.to_dict│                       │ → recorder.add_frame│
    └───────────────────────┘                       └──────────────────────┘
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

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


class CameraProcess:
    """Captures RealSense frames in a crash-isolated background process.

    Usage:
        cam = CameraProcess(CameraProcessConfig(serial="...", hz=30.0))
        cam.start()
        # In controller loop:
        frame = cam.poll_latest_frame()
        if frame is not None:
            recorder.add_frame(..., camera_frame=frame)
        cam.stop()
    """

    def __init__(self, config: CameraProcessConfig | None = None) -> None:
        self.config = config or CameraProcessConfig()
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()

        # maxsize=1 ensures only the latest frame is queued (old frames
        # dropped automatically when the queue is full).
        self._frame_queue: mp.Queue = mp.Queue(maxsize=1)

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
        self._drain_queue()

        self._process = mp.Process(
            target=self._run,
            name=f"camera-{self.config.camera_name}",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "CameraProcess started (name=%s, serial=%s, hz=%.0f)",
            self.config.camera_name, self.config.serial or "default",
            self.config.hz,
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
        self._drain_queue()
        logger.info("CameraProcess stopped.")

    # ------------------------------------------------------------------
    # Frame access (called from main process)
    # ------------------------------------------------------------------

    def poll_latest_frame(self) -> dict | None:
        """Non-blocking poll for the latest camera frame.

        Returns a dict (CameraFrame.to_dict() format) or None.
        Drains the queue to always return the newest frame.
        """
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
                "CameraProcess capture loop started @ %.0f Hz.", self.config.hz
            )

            last_ts = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    frame = cam.read(timeout_ms=self.config.timeout_ms)
                    frame_dict = frame.to_dict()

                    # Non-blocking put — drops oldest frame if queue is full
                    # (maxsize=1), ensuring we always keep the latest.
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

            cam.disconnect()
            logger.info("CameraProcess capture loop exited cleanly.")

        except (RuntimeError, OSError):
            logger.exception("CameraProcess crashed.")
            self._crashed.set()

    def _drain_queue(self) -> None:
        """Remove all pending frames from the queue."""
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

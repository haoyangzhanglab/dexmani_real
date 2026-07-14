"""Camera recording daemon process — crash-isolated frame capture.

Runs RealSense capture in a separate multiprocessing.Process so that USB
disconnects, firmware hangs, or frame timeouts don't crash the control loop.

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem).

Architecture:
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
import time
from dataclasses import dataclass

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class CameraProcessConfig:
    """Configuration for CameraProcess."""

    camera_name: str = "realsense"
    serial: str | None = None
    hz: float = 30.0
    warmup_frames: int = 10
    timeout_ms: int = 1000
    shm_name: str = "dexmani_cam_0"
    rgb_height: int = 480
    rgb_width: int = 640


class CameraProcess:
    """Captures RealSense frames in a crash-isolated background process.

    Frames are transported via CameraRingBuffer (zero-copy shared memory).

    Usage:
        cam = CameraProcess(CameraProcessConfig(serial="...", hz=30.0))
        cam.start()
        frame = cam.poll_latest_frame()  # reads from shm
        cam.stop()
    """

    def __init__(self, config: CameraProcessConfig | None = None) -> None:
        self.config = config or CameraProcessConfig()
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._shm_buf = None  # CameraRingBuffer instance (lazy init)
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
        self._init_shm()

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
        if self._shm_buf is not None:
            self._shm_buf.close()
            self._shm_buf.unlink()
            self._shm_buf = None
        logger.info("CameraProcess stopped.")

    # ------------------------------------------------------------------
    # Frame access (called from main process)
    # ------------------------------------------------------------------

    def poll_latest_frame(self) -> dict | None:
        """Non-blocking poll for the latest camera frame via shared memory."""
        return self._poll_shm()

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
                "CameraProcess capture loop started @ %.0f Hz.",
                self.config.hz,
            )

            from dexmani_real.shm.layouts import pack_camera_frame
            from dexmani_real.shm.ring_buffer import CameraRingBuffer

            h = self.config.rgb_height
            w = self.config.rgb_width
            shm_writer = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=False,
            )

            # A read that fails every iteration (align crash, USB drop, or the
            # L515 "no depth intrinsics" bad state) must not be swallowed forever
            # — otherwise the process stays alive producing zero frames with no
            # recovery. Count consecutive failures and, past a threshold, rebuild
            # the camera; RealSense.connect() self-heals the L515 via
            # hardware_reset, so recovery must happen here, in-process.
            reconnect_after = max(int(self.config.hz), 15)  # ~1 s of dead frames
            consecutive_failures = 0

            last_ts = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    frame = cam.read(timeout_ms=self.config.timeout_ms, compute_depth=False)
                    consecutive_failures = 0
                    try:
                        header, rgb, depth = pack_camera_frame(
                            frame.rgb, frame.depth_raw,
                            frame.timestamp, frame.frame_id,
                        )
                        shm_writer.write(header, rgb, depth)
                    except (ValueError, RuntimeError, OSError):
                        logger.exception(
                            "CameraProcess shm write failed — continuing."
                        )
                except (RuntimeError, OSError):
                    consecutive_failures += 1
                    if consecutive_failures % reconnect_after != 0:
                        logger.debug(
                            "CameraProcess frame read failed (%d) — continuing.",
                            consecutive_failures,
                        )
                    else:
                        logger.warning(
                            "CameraProcess: %d consecutive read failures — rebuilding camera.",
                            consecutive_failures,
                        )
                        cam.disconnect()
                        if cam.connect():
                            logger.info("CameraProcess camera rebuilt; resuming capture.")
                            consecutive_failures = 0
                        else:
                            logger.error("CameraProcess camera rebuild failed — will retry.")

                # Maintain target rate
                elapsed = time.monotonic() - last_ts
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_ts = time.monotonic()

            shm_writer.close()
            cam.disconnect()
            logger.info("CameraProcess capture loop exited cleanly.")

        except (RuntimeError, OSError):
            logger.exception("CameraProcess crashed.")
            self._crashed.set()

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
from dataclasses import dataclass, field

from dexmani_real.sensor.pointcloud_processor import PointCloudProcessorConfig
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
    # Raw depth stream resolution. 640×480 (= color resolution) avoids an
    # 8 ms validity-gate penalty on the L515: confidence/IR/edge checks run on
    # raw depth pre-alignment, so running both streams at the same resolution
    # cuts the gate pixel count by 2.56× with no accuracy loss — the aligned
    # output was already 640×480 under depth_to_color.
    # NOTE: D400-series cameras do not support 1024x768 depth — use (640, 480).
    depth_width: int = 640
    depth_height: int = 480
    # Apply the calibrated L515 image-domain validity gate (confidence/IR/edge,
    # realsense.L515_CALIBRATED_DEPTH_VALIDITY) to raw depth before alignment.
    # Affects the recorded /depth stream too (invalid pixels zeroed) — this is
    # the validated production behavior.
    depth_validity: bool = True
    # Compute a fixed-size world-frame pointcloud per frame in the child and
    # ship it through SHM (extrinsics resolved child-side from cameras.json by
    # the connected serial; eye-in-hand entries disable this with a warning).
    enable_pointcloud: bool = False
    pointcloud: PointCloudProcessorConfig = field(default_factory=PointCloudProcessorConfig)


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
        # Depth units in meters (L515: 0.00025) — set by the child after
        # camera connect; 0.0 means "not yet known".
        self._depth_scale = mp.Value("d", 0.0)
        # Hardware identity + intrinsics, set by the child after camera connect
        # so /meta is self-contained per episode (same pattern as depth_scale).
        self._camera_serial = mp.Array("c", 32)
        self._camera_K = mp.Array("d", 9)  # 3×3 intrinsics, flattened

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

    @property
    def depth_scale(self) -> float | None:
        """Raw uint16 depth units in meters, or None if not yet known."""
        v = self._depth_scale.value
        return v if v > 0 else None

    @property
    def pointcloud_meta(self) -> dict:
        """h5py-safe pointcloud processor params for /meta persistence."""
        if not self.config.enable_pointcloud:
            return {}
        return self.config.pointcloud.to_meta_dict()

    @property
    def camera_serial(self) -> str | None:
        """Hardware serial of the connected camera, or None if not yet known."""
        val = self._camera_serial.value
        if isinstance(val, bytes):
            s = val.rstrip(b"\x00").decode()
            return s if s else None
        return None

    @property
    def camera_K(self) -> np.ndarray | None:
        """3×3 camera intrinsics matrix from the connected hardware, or None."""
        import numpy as np

        v = np.array([self._camera_K[i] for i in range(9)], dtype=np.float64).reshape(3, 3)
        if np.all(v == 0):
            return None
        return v

    # ------------------------------------------------------------------
    # Shared memory helpers
    # ------------------------------------------------------------------

    def _init_shm(self) -> None:
        """Initialize or attach to the shared memory camera ring buffer."""
        from dexmani_real.shm.ring_buffer import CameraRingBuffer

        h = self.config.rgb_height
        w = self.config.rgb_width
        pc_shape = (self.config.pointcloud.num_points, 6) if self.config.enable_pointcloud else None
        try:
            self._shm_buf = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=True,
                pc_shape=pc_shape,
            )
        except FileExistsError:
            # Leftover block from a run that died without unlink. Its slot
            # geometry may differ (old layout / different pc_shape) and
            # attaching would silently corrupt frames — drop it and recreate.
            logger.warning(
                "CameraRingBuffer '%s' already exists (stale from a previous run) "
                "— unlinking and recreating.", self.config.shm_name,
            )
            from multiprocessing import shared_memory

            stale = shared_memory.SharedMemory(name=self.config.shm_name)
            stale.close()
            stale.unlink()
            self._shm_buf = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=True,
                pc_shape=pc_shape,
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
        header, rgb, depth, pointcloud, seq = result
        return bytes_to_camera_frame(header, rgb, depth, pointcloud=pointcloud)

    # ------------------------------------------------------------------
    # Internal (runs in child process)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main capture loop (runs in child process)."""
        interval = 1.0 / self.config.hz

        try:
            from dexmani_real.sensor.realsense import (
                L515_CALIBRATED_DEPTH_VALIDITY,
                RealSense,
                RealSenseConfig,
            )

            rs_config = RealSenseConfig(
                camera_name=self.config.camera_name,
                serial=self.config.serial,
                depth_resolution=(self.config.depth_width, self.config.depth_height),
                fps=int(self.config.hz),
                warmup_frames=self.config.warmup_frames,
                depth_validity=L515_CALIBRATED_DEPTH_VALIDITY if self.config.depth_validity else None,
            )

            cam = RealSense(rs_config)
            if not cam.connect():
                logger.error("CameraProcess: RealSense connect failed.")
                self._crashed.set()
                return
            self._depth_scale.value = cam.get_depth_scale()
            # Propagate hardware identity + intrinsics to the parent so /meta
            # is self-contained per episode (camera_K for offline pointcloud
            # regeneration; serial for extrinsics resolution).
            serial_str = str(cam.active_serial or "")
            self._camera_serial.value = (serial_str[:31] + "\x00").encode()
            if cam.K is not None:
                K_flat = cam.K.flatten()
                for i in range(9):
                    self._camera_K[i] = float(K_flat[i])

            processor = self._build_processor(cam) if self.config.enable_pointcloud else None

            logger.info(
                "CameraProcess capture loop started @ %.0f Hz.",
                self.config.hz,
            )

            import numpy as np

            from dexmani_real.shm.layouts import pack_camera_frame
            from dexmani_real.shm.ring_buffer import CameraRingBuffer

            h = self.config.rgb_height
            w = self.config.rgb_width
            pc_shape = (self.config.pointcloud.num_points, 6) if self.config.enable_pointcloud else None
            shm_writer = CameraRingBuffer(
                name=self.config.shm_name,
                rgb_shape=(h, w, 3),
                depth_shape=(h, w),
                maxlen=5,
                create=False,
                pc_shape=pc_shape,
            )

            # Empty-cloud fail-safe: re-send the last valid cloud; before the
            # first valid cloud, send zeros flagged pc_num_points=0.
            zero_pc = np.zeros(pc_shape, dtype=np.float32) if pc_shape else None
            last_valid_pc: np.ndarray | None = None
            empty_streak = 0
            empty_warn_every = max(int(self.config.hz), 1)  # ~1 s

            # A read that fails every iteration (align crash, USB drop, or the
            # L515 "no depth intrinsics" bad state) must not be swallowed forever
            # — otherwise the process stays alive producing zero frames with no
            # recovery. A failure *count* has no consistent wall-clock meaning
            # (a timeout blocks ~timeout_ms, an align crash fails in ~1 ms), so
            # track time since the last good read (or last recovery attempt)
            # and rebuild the camera past a threshold; RealSense.connect()
            # self-heals the L515 via hardware_reset, so recovery must happen
            # here, in-process.
            stall_rebuild_s = 4.0
            last_ok = time.monotonic()

            last_ts = time.monotonic()
            # ── timing instrumentation: log step breakdown every ~1s ──
            t_read = t_pc = t_pack = 0.0
            timing_window = max(int(self.config.hz), 1)
            timing_n = 0
            while not self._stop_event.is_set():
                try:
                    t0 = time.monotonic()
                    frame = cam.read(
                        timeout_ms=self.config.timeout_ms,
                        compute_depth=processor is not None,
                    )
                    t1 = time.monotonic()
                    last_ok = t1
                    pc = None
                    if processor is not None:
                        try:
                            pc = processor.process(frame.depth, frame.rgb, cam.get_rays())
                        except (RuntimeError, ValueError):
                            logger.exception("CameraProcess pointcloud processing failed.")
                        if pc is not None:
                            last_valid_pc = pc
                            empty_streak = 0
                        else:
                            pc = last_valid_pc  # None until the first valid cloud
                            empty_streak += 1
                            if empty_streak == 1 or empty_streak % empty_warn_every == 0:
                                logger.warning(
                                    "CameraProcess: empty pointcloud (%d consecutive) — "
                                    "%s.", empty_streak,
                                    "re-sending last valid" if pc is not None else "sending zeros",
                                )
                    t2 = time.monotonic()
                    try:
                        header, rgb, depth = pack_camera_frame(
                            frame.rgb, frame.depth_raw,
                            frame.timestamp, frame.frame_id,
                            pc_num_points=pc.shape[0] if pc is not None else 0,
                        )
                        shm_writer.write(
                            header, rgb, depth,
                            pointcloud=pc if pc is not None else zero_pc,
                        )
                    except (ValueError, RuntimeError, OSError):
                        logger.exception(
                            "CameraProcess shm write failed — continuing."
                        )
                    t3 = time.monotonic()

                    # Accumulate timing for periodic breakdown log
                    t_read += (t1 - t0) * 1000
                    t_pc += (t2 - t1) * 1000
                    t_pack += (t3 - t2) * 1000
                    timing_n += 1
                    if timing_n >= timing_window:
                        logger.debug(
                            "CameraProcess timing [%d frames]: "
                            "read=%.1fms pc=%.1fms pack+shm=%.1fms total=%.0fms",
                            timing_n, t_read / timing_n, t_pc / timing_n, t_pack / timing_n,
                            (t_read + t_pc + t_pack) / timing_n,
                        )
                        t_read = t_pc = t_pack = 0.0
                        timing_n = 0
                except (RuntimeError, OSError):
                    stalled_s = time.monotonic() - last_ok
                    if stalled_s <= stall_rebuild_s:
                        logger.debug(
                            "CameraProcess frame read failed (%.1fs since last frame) — continuing.",
                            stalled_s,
                        )
                    else:
                        logger.warning(
                            "CameraProcess: no frame for %.1fs — rebuilding camera.",
                            stalled_s,
                        )
                        cam.disconnect()
                        if cam.connect():
                            logger.info("CameraProcess camera rebuilt; resuming capture.")
                            self._depth_scale.value = cam.get_depth_scale()
                            if cam.K is not None:
                                K_flat = cam.K.flatten()
                                for i in range(9):
                                    self._camera_K[i] = float(K_flat[i])
                        else:
                            logger.error("CameraProcess camera rebuild failed — will retry.")
                        # Reset regardless of outcome so failed rebuilds retry
                        # paced ≥stall_rebuild_s apart (no hardware_reset hammering).
                        last_ok = time.monotonic()

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

    def _build_processor(self, cam):
        """Resolve extrinsics from cameras.json by the connected serial and
        build the PointCloudProcessor (child process only).

        Any failure (missing entry, eye-in-hand without per-frame T_base_eef,
        unreadable cameras.json) disables the pointcloud with a loud warning —
        rgb/depth capture continues; the SHM slot keeps zeros/pc_num_points=0.
        """
        try:
            from dexmani_real.config.camera_calib import CameraCalib
            from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor

            calib = CameraCalib()
            cam_name = calib.resolve_name_by_serial(str(cam.active_serial))
            T_world_camera = calib.get_extrinsics(cam_name)  # ValueError if eye_in_hand
            processor = PointCloudProcessor(T_world_camera, self.config.pointcloud)
            logger.info(
                "CameraProcess pointcloud enabled: %s (eye-to-hand), T pos=%s",
                cam_name, T_world_camera[:3, 3].round(3).tolist(),
            )
            return processor
        except (KeyError, ValueError, FileNotFoundError, OSError):
            logger.exception(
                "CameraProcess: pointcloud DISABLED — calibration unresolved "
                "(missing entry / eye-in-hand / invalid cameras.json)."
            )
            return None

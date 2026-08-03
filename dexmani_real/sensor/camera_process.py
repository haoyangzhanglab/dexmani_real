"""Camera recording daemon process — crash-isolated frame capture.

Runs RealSense capture in a separate multiprocessing.Process so that USB
disconnects, firmware hangs, or frame timeouts don't crash the control loop.

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem).

Architecture:
    ┌───────────────────────┐   SharedMemory         ┌──────────────────────┐
    │ CameraProcess         │ ── CameraRingBuffer ──►│ TeleopController     │
    │ (独立 mp.Process)     │   (zero-copy)          │ (主进程, 16Hz)       │
    │                       │                        │                      │
    │ RealSense.read()      │                        │ poll_latest_frame()  │
    │ → shm.write()         │                        │ → shm.read_latest()  │
    └───────────────────────┘                        └──────────────────────┘
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from dexmani_real.sensor.pointcloud_processor import PointCloudProcessorConfig
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedString

    import numpy as np

    from dexmani_real.sensor.protocols import CameraDriver
    from dexmani_real.shm.ring_buffer import CameraRingBuffer
    from dexmani_real.shm.shared_storage import SharedStorage

logger = get_logger(__name__)


@dataclass
class CameraProcessConfig:
    """Configuration for CameraProcess."""

    camera_name: str = "realsense"
    serial: str | None = None
    hz: float = 30.0
    warmup_frames: int = 10
    # Per-frame wait_for_frames timeout (ms).  L515 @ 30 Hz delivers every ~33 ms;
    # 300 ms allows 9 consecutive misses before a timeout exception — generous for
    # transient glitches while keeping shutdown detection responsive (< 0.3 s worst
    # case vs the parent's 3.0 s join deadline).
    timeout_ms: int = 300
    shm_name: str = "dexmani_cam_0"
    rgb_height: int = 480
    rgb_width: int = 640
    # Raw depth stream resolution. 640×480 (= color resolution, same as aligned
    # output). NOTE: D400-series cameras do not support 1024x768 depth.
    depth_width: int = 640
    depth_height: int = 480
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

    def __init__(
        self,
        config: CameraProcessConfig | None = None,
        camera_factory: Callable[[CameraProcessConfig], CameraDriver] | None = None,
    ) -> None:
        self.config = config or CameraProcessConfig()
        self._camera_factory = camera_factory  # None → use RealSense default
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._shm_buf: CameraRingBuffer | None = None  # CameraRingBuffer instance (lazy init)
        self._crashed = mp.Event()
        # Depth units in meters (L515: 0.00025) — set by the child after
        # camera connect; 0.0 means "not yet known".
        self._depth_scale = mp.Value("d", 0.0)
        # Hardware identity + intrinsics, set by the child after camera connect
        # so /meta is self-contained per episode (same pattern as depth_scale).
        # mp.Array("c", ...) is SynchronizedString at runtime (has .value/.raw);
        # typeshed's str overload yields SynchronizedArray[Any] (no .value) — hence the ignore.
        self._camera_serial: SynchronizedString = mp.Array("c", 32)  # type: ignore[assignment]  # typeshed gap: str overload -> SynchronizedArray[Any], runtime type is SynchronizedString
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
            self.config.camera_name,
            self.config.serial or "default",
            self.config.hz,
        )
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop and wait for process exit."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("CameraProcess did not exit within %.1fs, terminating.", timeout)
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
                "CameraRingBuffer '%s' already exists (stale from a previous run) " "— unlinking and recreating.",
                self.config.shm_name,
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

        # ── 线程池限制 ──
        # CameraProcess 作为独立子进程, OpenCV/NumPy 在多数核心机器上默认
        # 各自派生子线程, 争抢 16Hz 控制循环的 CPU 时间片。限制为单线程,
        # 依赖进程级并行 (arm/hand/camera 各自独立进程) 而非每库内部多线程展开。
        try:
            import cv2

            cv2.setNumThreads(1)
        except ImportError:
            pass

        try:
            if self._camera_factory is not None:
                cam = self._camera_factory(self.config)
            else:
                from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

                rs_config = RealSenseConfig(
                    camera_name=self.config.camera_name,
                    serial=self.config.serial,
                    depth_resolution=(self.config.depth_width, self.config.depth_height),
                    fps=int(self.config.hz),
                    warmup_frames=self.config.warmup_frames,
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

            from dexmani_real.shm.ring_buffer import CameraRingBuffer

            shm_writer = CameraRingBuffer.attach(self.config.shm_name)

            pc_shape = (self.config.pointcloud.num_points, 6) if self.config.enable_pointcloud else None

            # Empty-cloud fail-safe: re-send the last valid cloud; before the
            # first valid cloud, send zeros flagged pc_num_points=0.
            zero_pc = np.zeros(pc_shape, dtype=np.float32) if pc_shape else None
            last_valid_pc: np.ndarray | None = None
            empty_streak = 0
            empty_warn_every = max(int(self.config.hz), 1)  # ~1 s

            interval: float = 1.0 / self.config.hz
            last_ts: float = time.monotonic()

            while not self._stop_event.is_set():
                try:
                    frame = cam.read(
                        timeout_ms=self.config.timeout_ms,
                        compute_depth=processor is not None,
                    )
                    # Mid-iteration stop check: after a potentially blocking
                    # cam.read() (up to timeout_ms ms), re-check the stop event
                    # before spending time on pointcloud computation and SHM
                    # writes that will be thrown away.
                    if self._stop_event.is_set():
                        break
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
                                    "CameraProcess: empty pointcloud (%d consecutive) — " "%s.",
                                    empty_streak,
                                    "re-sending last valid" if pc is not None else "sending zeros",
                                )
                    try:
                        header, rgb, depth = pack_camera_frame(
                            frame.rgb,  # type: ignore[arg-type]
                            frame.depth_raw,
                            frame.timestamp,
                            frame.frame_id,
                            pc_num_points=pc.shape[0] if pc is not None else 0,
                            camera_health=0,
                        )
                        shm_writer.write(
                            header,
                            rgb,
                            depth,
                            pointcloud=pc if pc is not None else zero_pc,
                        )
                    except (ValueError, RuntimeError, OSError):
                        logger.exception("CameraProcess shm write failed — continuing.")
                except (RuntimeError, OSError):
                    logger.debug("CameraProcess frame read failed — continuing.")

                # Maintain target rate
                elapsed = time.monotonic() - last_ts
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_ts = time.monotonic()

            # Close SHM writer before camera disconnect — shm_writer.close()
            # is a fast kernel call that never blocks, whereas cam.disconnect()
            # (→ pipeline.stop()) can block indefinitely when the L515 USB
            # device is in a bad state (mid-run stream stall, USB starvation).
            # Releasing the child's SHM handle first ensures the parent's
            # unlink() sees zero open handles regardless of disconnect timing.
            shm_writer.close()
            try:
                cam.disconnect()
            except Exception:
                pass
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
                cam_name,
                T_world_camera[:3, 3].round(3).tolist(),
            )
            return processor
        except (KeyError, ValueError, FileNotFoundError, OSError):
            logger.exception(
                "CameraProcess: pointcloud DISABLED — calibration unresolved "
                "(missing entry / eye-in-hand / invalid cameras.json)."
            )
            return None


# ------------------------------------------------------------------
# CameraSession — bundled lifecycle for entry points
# ------------------------------------------------------------------


@dataclass
class CameraSession:
    """CameraProcess + calibration + serial resolution, bundled for entry points.

    Every real entry point duplicates ~40 lines of camera setup (create
    CameraProcess, start, load CameraCalib, define a _resolve_camera_name
    closure).  This dataclass and :func:`create_camera_session` collapse
    that into a single call.

    ``camera`` is None when the camera process failed to start — all
    accessors return safe defaults so the entry point can degrade
    gracefully (no camera frames, no /meta extrinsics).
    """

    camera: CameraProcess | None
    calib: object | None  # CameraCalib — lazy import to stay lightweight
    _name_cache: str | None = None

    def resolve_name(self) -> str | None:
        """serial → cameras.json entry name, cached after first resolution.

        Returns None until the child process finishes connect() and the
        serial becomes available.
        """
        if self._name_cache is not None or self.calib is None or self.camera is None:
            return self._name_cache
        ser = self.camera.camera_serial
        if not ser:
            return None
        try:
            self._name_cache = self.calib.resolve_name_by_serial(ser)  # type: ignore[attr-defined,union-attr]
        except KeyError:
            pass
        return self._name_cache

    # -- delegated properties (safe when camera is None) --

    @property
    def crashed(self) -> bool:
        return self.camera is not None and self.camera.crashed

    @property
    def depth_scale(self) -> float | None:
        return self.camera.depth_scale if self.camera is not None else None

    @property
    def camera_K(self) -> "np.ndarray | None":
        return self.camera.camera_K if self.camera is not None else None

    @property
    def pointcloud_meta(self) -> dict:
        if self.camera is not None:
            return dict(self.camera.pointcloud_meta)
        return {}

    def poll_latest_frame(self) -> dict | None:
        return self.camera.poll_latest_frame() if self.camera is not None else None

    def stop(self) -> None:
        if self.camera is not None:
            self.camera.stop()


def create_camera_session(enable_pointcloud: bool = True, hz: float = 30.0) -> CameraSession:
    """Create, start, and calibrate a CameraProcess in one call.

    Returns a CameraSession whose ``camera`` is None when the process
    failed to start — entry points degrade gracefully (no images, no
    /meta extrinsics).
    """
    from dexmani_real.config.camera_calib import CameraCalib

    camera = CameraProcess(CameraProcessConfig(camera_name="realsense", hz=hz, enable_pointcloud=enable_pointcloud))
    if camera.start():
        print("Camera 进程已启动 (RealSense @30Hz, SHM, pointcloud)")
    else:
        print("Camera 启动失败 (降级: 只录关节/EEF, 不录图像)")
        return CameraSession(camera=None, calib=None)

    calib = None
    try:
        calib = CameraCalib()
    except (OSError, ValueError, KeyError):
        print("cameras.json 加载失败 — /meta 将缺少外参（点云不受影响，子进程独立解析）")

    return CameraSession(camera=camera, calib=calib)


# ═══════════════════════════════════════════════════════════════════
# Camera frame packing helpers (moved from shm/layouts.py — Phase 2.7)
# ═══════════════════════════════════════════════════════════════════


def pack_camera_frame(
    rgb: "np.ndarray",
    depth_raw: "np.ndarray",
    timestamp: float,
    frame_id: int,
    pc_num_points: int = 0,
    camera_health: int = 0,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Pack camera frame attributes into (header, rgb_bytes, depth_bytes)."""
    import numpy as np

    from dexmani_real.shm.ring_buffer import CAMERA_FRAME_HEADER_DTYPE

    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["timestamp"] = np.float64(timestamp)
    header["frame_number"] = np.uint64(frame_id)
    header["pc_num_points"] = np.uint32(pc_num_points)
    header["camera_health"] = np.uint8(camera_health)

    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    depth_arr = np.asarray(depth_raw, dtype=np.uint16)

    header["rgb_size"] = np.uint64(rgb_arr.nbytes)
    header["depth_size"] = np.uint64(depth_arr.nbytes)
    header["rgb_shape_h"] = np.uint32(rgb_arr.shape[0])
    header["rgb_shape_w"] = np.uint32(rgb_arr.shape[1])
    header["rgb_shape_c"] = np.uint32(rgb_arr.shape[2])
    header["depth_shape_h"] = np.uint32(depth_arr.shape[0])
    header["depth_shape_w"] = np.uint32(depth_arr.shape[1])

    return header, rgb_arr, depth_arr


def bytes_to_camera_frame(
    header: "np.ndarray",
    rgb_bytes: "np.ndarray",
    depth_bytes: "np.ndarray",
    pointcloud: "np.ndarray | None" = None,
) -> dict:
    """Reconstruct a camera frame dict from raw bytes."""
    h = header[0]
    rgb = rgb_bytes.reshape((int(h["rgb_shape_h"]), int(h["rgb_shape_w"]), int(h["rgb_shape_c"]))).copy()
    depth = depth_bytes.reshape((int(h["depth_shape_h"]), int(h["depth_shape_w"]))).copy()
    frame = {
        "rgb": rgb,
        "depth": depth,
        "timestamp": float(h["timestamp"]),
        "frame_number": int(h["frame_number"]),
        "camera_health": int(h["camera_health"]),
    }
    if pointcloud is not None:
        frame["pointcloud"] = pointcloud
        frame["pointcloud_valid"] = int(h["pc_num_points"]) > 0
    return frame


# ═══════════════════════════════════════════════════════════════════
# New architecture: camera_loop (mp.Process target)
# ═══════════════════════════════════════════════════════════════════


def camera_loop(shared: "SharedStorage") -> None:
    """Run RealSense camera → bridge frames to shared.camera_ring.

    Designed as an mp.Process target. Uses :func:`create_camera_session` for
    lifecycle and :func:`pack_camera_frame` for SHM encoding.

    On init failure, logs the error and returns without setting
    ``shared.camera_ready`` — Main detects this via ready-event timeout.
    """
    _logger = get_logger("camera_loop")

    session = create_camera_session()
    if session.camera is None:
        _logger.error("camera_loop: camera init failed")
        return

    # Publish camera metadata for Policy to read before recording starts.
    _ds = session.depth_scale
    if _ds is not None:
        shared.camera_depth_scale.value = float(_ds)
    _K = session.camera_K
    if _K is not None and _K.shape == (3, 3):
        shared.camera_K[:] = _K.flatten().tolist()
    _serial = getattr(session.camera, "camera_serial", "")
    if _serial:
        shared.camera_serial.value = _serial.encode()[:31].ljust(32, b"\x00")

    shared.camera_ready.set()
    _logger.info("Camera loop ready")

    while shared.is_running.value:
        shared.camera_heartbeat_s.value = time.monotonic()
        frame = session.poll_latest_frame()
        if frame is not None and shared.is_recording.value:
            try:
                pc = frame.get("pointcloud")
                pc_num = pc.shape[0] if pc is not None and frame.get("pointcloud_valid", False) else 0
                header, rgb, depth = pack_camera_frame(
                    frame["rgb"], frame["depth"], frame["timestamp"],
                    frame["frame_number"], pc_num,
                    frame.get("camera_health", 0),
                )
                shared.camera_ring.write(header, rgb, depth, pc)
            except Exception:
                _logger.warning("camera_loop: bridge write failed", exc_info=True)
        time.sleep(0.03)  # ~30Hz

    session.stop()
    _logger.info("Camera loop exited")

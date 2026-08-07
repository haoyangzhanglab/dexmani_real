"""Camera process target for SharedStorage architecture.

Runs RealSense capture as an ``mp.Process`` target, writing frames directly to
``shared.camera_ring`` via :class:`~dexmani_real.shm.ring_buffer.CameraRingBuffer`.

Architecture (SharedStorage — 5-process model):
    camera_loop writes → shared.camera_ring → policy_loop reads (single-clock recording)

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from dexmani_real.config.defaults import policy
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

if TYPE_CHECKING:
    import numpy as np

    from dexmani_real.shm.shared_storage import SharedStorage

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Camera frame packing helper
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


# ═══════════════════════════════════════════════════════════════════
# camera_loop — mp.Process target
# ═══════════════════════════════════════════════════════════════════


def camera_loop(shared: "SharedStorage") -> None:
    """Run RealSense camera → write frames directly to ``shared.camera_ring``.

    Designed as an ``mp.Process`` target. Runs the camera capture loop directly
    in this process — no subprocess spawn.

    On init failure, logs the error and returns without setting
    ``shared.camera_ready`` — Main detects this via ready-event timeout.
    """
    import numpy as np

    _logger = get_logger("camera_loop")

    # ── Thread pool limit ──
    # OpenCV/NumPy default to multi-threading on many-core machines, competing
    # for CPU with the 16 Hz control loop.  We rely on process-level parallelism
    # (arm/hand/camera each in its own process), not per-library thread pools.
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    cam = None
    try:
        # ── Create and connect camera ──
        from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

        rs_config = RealSenseConfig(
            camera_name="realsense",
            depth_resolution=(640, 480),
            fps=30,
            warmup_frames=10,
        )
        cam = RealSense(rs_config)

        if not cam.connect():
            _logger.error("camera_loop: RealSense connect failed")
            return

        # ── Publish metadata to SharedStorage ──
        shared.camera_depth_scale.value = float(cam.get_depth_scale())
        _serial_raw = str(cam.active_serial or "")
        shared.camera_serial.value = _serial_raw[:31].ljust(32, "\x00").encode()
        if cam.K is not None:
            shared.camera_K[:] = cam.K.flatten().tolist()

        # ── Build pointcloud processor (best-effort) ──
        processor: Any = None  # PointCloudProcessor when enabled
        pc_shape: tuple[int, int] | None = None
        try:
            from dexmani_real.config.camera_calib import CameraCalib
            from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig

            calib = CameraCalib()
            cam_name = calib.resolve_name_by_serial(str(cam.active_serial))
            T_world_camera = calib.get_extrinsics(cam_name)
            pc_config = PointCloudProcessorConfig()
            processor = PointCloudProcessor(T_world_camera, pc_config)
            pc_shape = (pc_config.num_points, 6)
            _logger.info(
                "camera_loop: pointcloud enabled, T pos=%s",
                T_world_camera[:3, 3].round(3).tolist(),
            )
        except Exception:
            _logger.warning("camera_loop: pointcloud DISABLED", exc_info=True)

        # Zero pointcloud fallback — used when process() returns None during recording.
        zero_pc = np.zeros(pc_shape, dtype=np.float32) if pc_shape else None

        shared.camera_ready.set()
        _logger.info("camera_loop: ready @ %.1f Hz (from policy.control_hz)", policy.control_hz)

        # ── Main capture loop ──
        rate_mgr = RateManager(policy.control_hz)

        # Forward-fill cache: when is_recording transitions False→True, the
        # first ring write uses the most recently computed pointcloud so the
        # policy never sees a frame with missing pc data (~1 tick old is fine).
        _last_pc: np.ndarray | None = None
        _was_recording: bool = False

        while shared.is_running.value:
            shared.camera_heartbeat_s.value = time.monotonic()

            # --- read frame ---
            try:
                frame = cam.read(timeout_ms=300, compute_depth=processor is not None)
            except (RuntimeError, OSError):
                _logger.warning("camera_loop: frame read failed", exc_info=True)
                # Maintain target rate even on read failure so a persistent
                # error doesn't turn into a tight loop.
                rate_mgr.wait()
                continue

            # --- pointcloud (only when recording) ---
            # Pointcloud processing (~46 ms) is the dominant cost in the pipeline.
            # Computing it every tick — even when no one consumes the data — would
            # needlessly burn CPU.  A forward-fill cache (_last_pc) bridges the
            # cross-process delay between the Policy setting is_recording=True and
            # the camera loop observing it — the first recorded frame gets the
            # most recently computed pc (~1 tick old).
            pc: np.ndarray | None = None
            _recording_now: bool = shared.is_recording.value

            # --- write to SharedStorage ring ---
            # Only bridge frames when Policy is recording — avoids sustained SHM
            # writes (~1.6 MB/frame) when nobody is consuming them.
            if _recording_now:
                if processor is not None:
                    # On the first tick after is_recording flips True, forward-fill
                    # the cached pointcloud to avoid a gap.
                    if not _was_recording and _last_pc is not None:
                        pc = _last_pc
                    else:
                        try:
                            pc = processor.process(frame.depth, frame.rgb, cam.get_rays())
                        except Exception:
                            _logger.warning("camera_loop: pointcloud processing failed", exc_info=True)
                try:
                    header, rgb, depth = pack_camera_frame(
                        frame.rgb,  # type: ignore[arg-type]
                        frame.depth_raw,
                        frame.timestamp,
                        frame.frame_id,
                        pc_num_points=pc.shape[0] if pc is not None else 0,
                        camera_health=0,
                    )
                    shared.camera_ring.write(
                        header,
                        rgb,
                        depth,
                        pointcloud=pc if pc is not None else zero_pc,
                    )
                except Exception:
                    _logger.warning("camera_loop: ring write failed", exc_info=True)

                # Update forward-fill cache (only when recording — non-recording
                # ticks don't compute pc, so the cache stays fresh from the last
                # recording tick).
                if pc is not None:
                    _last_pc = pc

            _was_recording = _recording_now

            # --- maintain target rate (absolute-deadline scheduling, consistent with other loops) ---
            rate_mgr.wait()

    except Exception:
        _logger.exception("camera_loop: crashed")
    finally:
        if cam is not None:
            try:
                cam.disconnect()
            except Exception:
                pass
        _logger.info("camera_loop: exited")

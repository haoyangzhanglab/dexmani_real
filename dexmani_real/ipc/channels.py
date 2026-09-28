"""Shared-memory sensor rings, events, and flags for runtime processes."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    VR_FRAME_DTYPE,
    make_pointcloud_frame_dtype,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Wire value of runtime.safety.SafetyState.DISARMED.
DISARMED_SAFETY_STATE_WIRE_VALUE = 0


@dataclass
class RuntimeChannelsConfig:
    """Sensor ring capacities and camera resolution defaults."""

    camera_ring_maxlen: int = CameraParams.ring_maxlen
    vr_ring_maxlen: int = 8
    pointcloud_num_points: int = 1024
    pointcloud_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: CameraParams().rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: CameraParams().depth_shape)

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.pointcloud_ring_maxlen,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("RuntimeChannels ring capacities must be positive")
        if (
            isinstance(self.pointcloud_num_points, bool)
            or not isinstance(self.pointcloud_num_points, (int, np.integer))
            or self.pointcloud_num_points <= 0
        ):
            raise ValueError("RuntimeChannels pointcloud_num_points must be a positive integer")

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        pointcloud_num_points: int | None = None,
    ) -> "RuntimeChannelsConfig":
        cam = getattr(runtime, "camera")
        return cls(
            camera_ring_maxlen=int(cam.ring_maxlen),
            camera_rgb_shape=(int(cam.height), int(cam.width), 3),
            camera_depth_shape=(int(cam.height), int(cam.width)),
            pointcloud_num_points=(
                runtime.pointcloud.num_points
                if pointcloud_num_points is None
                else pointcloud_num_points
            ),
        )


_RING_RESOURCE_NAMES = (
    "camera_ring",
    "vr_ring",
    "pointcloud_ring",
)


@dataclass
class RuntimeChannels:
    """Runtime channels created in Main before spawning child processes."""

    camera_ring: CameraRingBuffer  # camera -> local observation consumers
    vr_ring: SharedMemoryRingBuffer  # VR -> local teleop/calibration
    pointcloud_ring: SharedMemoryRingBuffer  # pointcloud worker -> local policy

    run_id: Any  # advancing the epoch invalidates pending motion commands
    # Latest software RUNNING termination; written under motion_lock.
    run_ended_reason: Any

    is_running: Any  # session lifetime, shared with sensors
    # Sticky runtime/safety fault: supervision takes the FAULT shutdown path.
    error_state: Any
    estop_request: Any  # sticky emergency-stop request
    quit_requested: Any  # operator or episode budget requests session exit
    start_request: Any  # operator -> local policy runner: B
    # Operator -> local policy runner: true only after the I/O owner completed the
    # authorized hand-home + collision-checked arm-home sequence.
    physical_home_completed: Any
    # Operator -> local policy runner: S request.
    stop_request: Any

    safety_state: Any  # SafetyState enum (0-3), guarded by motion_lock
    # Serializes motion permissions; never held across hardware SDK calls.
    motion_lock: Any

    vr_ready: Any
    camera_ready: Any
    pointcloud_ready: Any

    camera_depth_scale: Any  # metres per raw depth unit, reported by the camera
    camera_serial: Any  # serial number string
    camera_geometry: Any  # static native RGB-D geometry JSON
    _closed: bool = field(init=False, repr=False, default=False)

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: RuntimeChannelsConfig | None = None,
        mp_context: Any | None = None,
    ) -> "RuntimeChannels":
        """Create sensor rings, flags, and events.

        Call once from Main before spawning child processes.
        """
        cfg = config or RuntimeChannelsConfig()
        ctx = mp_context or mp.get_context("spawn")

        storage = cls.__new__(cls)
        storage._closed = False
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx)
        except BaseException as allocation_error:
            try:
                cleanup_succeeded = storage.close()
            except BaseException:
                logger.critical("RuntimeChannels allocation rollback raised", exc_info=True)
                raise RuntimeError(
                    "RuntimeChannels allocation failed and rollback raised"
                ) from allocation_error
            if not cleanup_succeeded:
                raise RuntimeError(
                    "RuntimeChannels allocation failed and rollback was incomplete"
                ) from allocation_error
            raise

        logger.debug("RuntimeChannels created (prefix=%s)", prefix)
        return storage

    @staticmethod
    def _allocate_resources(
        storage: "RuntimeChannels",
        prefix: str,
        cfg: RuntimeChannelsConfig,
        ctx: Any,
    ) -> None:
        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=cfg.camera_rgb_shape,
            depth_shape=cfg.camera_depth_shape,
            maxlen=cfg.camera_ring_maxlen,
            create=True,
        )
        storage.vr_ring = SharedMemoryRingBuffer(
            f"{prefix}_vr",
            dtype=VR_FRAME_DTYPE,
            maxlen=cfg.vr_ring_maxlen,
            create=True,
        )
        storage.pointcloud_ring = SharedMemoryRingBuffer(
            f"{prefix}_pointcloud",
            dtype=make_pointcloud_frame_dtype(cfg.pointcloud_num_points),
            maxlen=cfg.pointcloud_ring_maxlen,
            create=True,
        )

        storage.run_id = ctx.Value("Q", 1)
        storage.run_ended_reason = ctx.Value("i", 0)

        storage.is_running = ctx.Value("b", True, lock=False)
        storage.error_state = ctx.Value("b", False)
        storage.estop_request = ctx.Value("b", False)
        storage.quit_requested = ctx.Value("b", False)
        storage.start_request = ctx.Value("b", False)
        storage.physical_home_completed = ctx.Value("b", False)
        storage.stop_request = ctx.Value("b", False)

        storage.safety_state = ctx.Value("i", DISARMED_SAFETY_STATE_WIRE_VALUE)
        storage.motion_lock = ctx.RLock()

        storage.vr_ready = ctx.Event()
        storage.camera_ready = ctx.Event()
        storage.pointcloud_ready = ctx.Event()

        storage.camera_depth_scale = ctx.Value("d", 0.0, lock=False)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32, lock=False)
        storage.camera_geometry = ctx.Array("c", b"\x00" * 2048, lock=False)

    def close(self) -> bool:
        """Close and unlink shared memory, attempting every resource even after failures.

        An already-unlinked segment raises ``FileNotFoundError``; close calls are
        idempotent, so cleanup can be retried. Return whether every resource was
        closed and unlinked successfully.
        """
        if bool(getattr(self, "_closed", False)):
            return True

        errors: list[str] = []

        def _attempt(operation: str, callback: Any, *, missing_ok: bool = False) -> bool:
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    errors.append(operation)
                    logger.warning("RuntimeChannels close: %s failed", operation, exc_info=True)
                    return False
            except Exception:
                errors.append(operation)
                logger.warning("RuntimeChannels close: %s failed", operation, exc_info=True)
                return False
            return True

        for ring_name in _RING_RESOURCE_NAMES:
            ring = getattr(self, ring_name, None)
            if ring is None:
                continue
            _attempt(f"{ring_name}.close", ring.close)
            _attempt(f"{ring_name}.unlink", ring.unlink, missing_ok=True)

        self._closed = not errors
        if self._closed:
            logger.debug("RuntimeChannels closed cleanly")
        else:
            logger.error("RuntimeChannels close incomplete: %s", ", ".join(errors))
        return self._closed

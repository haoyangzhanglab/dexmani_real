"""Shared-memory sensor rings, events, and flags for runtime processes."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    VR_FRAME_DTYPE,
    make_camera_frame_dtype,
    make_pointcloud_frame_dtype,
)


logger = logging.getLogger(__name__)


@dataclass
class SensorChannelsConfig:
    """Sensor ring capacities and camera resolution defaults."""

    camera: bool = False
    vr: bool = False
    pointcloud: bool = False

    camera_ring_maxlen: int = CameraParams.ring_maxlen
    vr_ring_maxlen: int = 8
    pointcloud_num_points: int = 1024
    pointcloud_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: CameraParams().rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: CameraParams().depth_shape)

    def __post_init__(self) -> None:
        if self.pointcloud and not self.camera:
            raise ValueError("pointcloud requires camera storage")
        capacities = tuple(
            value
            for enabled, value in (
                (self.camera, self.camera_ring_maxlen),
                (self.vr, self.vr_ring_maxlen),
                (self.pointcloud, self.pointcloud_ring_maxlen),
            )
            if enabled
        )
        if any(type(value) is not int or value <= 0 for value in capacities):
            raise ValueError("sensor ring capacities must be positive integers")
        if self.pointcloud and (
            isinstance(self.pointcloud_num_points, bool)
            or not isinstance(self.pointcloud_num_points, (int, np.integer))
            or self.pointcloud_num_points <= 0
        ):
            raise ValueError("SensorChannels pointcloud_num_points must be a positive integer")

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        camera: bool = False,
        vr: bool = False,
        pointcloud: bool = False,
    ) -> "SensorChannelsConfig":
        cam = runtime.camera
        return cls(
            camera=camera,
            vr=vr,
            pointcloud=pointcloud,
            camera_ring_maxlen=cam.ring_maxlen,
            camera_rgb_shape=cam.rgb_shape,
            camera_depth_shape=cam.depth_shape,
            pointcloud_num_points=runtime.pointcloud.num_points,
        )


_RING_RESOURCE_NAMES = (
    "camera_ring",
    "vr_ring",
    "pointcloud_ring",
)


@dataclass
class SensorChannels:
    """Main-owned sensor IPC; workers attach without unlinking."""

    camera_ring: SharedMemoryRingBuffer | None  # camera -> local observation consumers
    vr_ring: SharedMemoryRingBuffer | None  # VR -> local teleop/calibration
    pointcloud_ring: SharedMemoryRingBuffer | None  # pointcloud worker -> local policy

    is_running: Any  # session lifetime, shared with sensor workers

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
        config: SensorChannelsConfig | None = None,
        mp_context: Any | None = None,
    ) -> "SensorChannels":
        """Allocate before spawn, rolling back partially allocated rings on failure."""
        cfg = config or SensorChannelsConfig()
        ctx = mp_context or mp.get_context("spawn")

        storage = cls.__new__(cls)
        storage._closed = False
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx)
        except BaseException as allocation_error:
            try:
                cleanup_succeeded = storage.close()
            except BaseException:
                logger.critical("SensorChannels allocation rollback raised", exc_info=True)
                raise RuntimeError(
                    "SensorChannels allocation failed and rollback raised"
                ) from allocation_error
            if not cleanup_succeeded:
                raise RuntimeError(
                    "SensorChannels allocation failed and rollback was incomplete"
                ) from allocation_error
            raise

        logger.debug("SensorChannels created (prefix=%s)", prefix)
        return storage

    @staticmethod
    def _allocate_resources(
        storage: "SensorChannels",
        prefix: str,
        cfg: SensorChannelsConfig,
        ctx: Any,
    ) -> None:
        storage.camera_ring = (
            SharedMemoryRingBuffer(
                name=f"{prefix}_camera",
                dtype=make_camera_frame_dtype(cfg.camera_rgb_shape, cfg.camera_depth_shape),
                maxlen=cfg.camera_ring_maxlen,
                create=True,
            )
            if cfg.camera
            else None
        )
        storage.vr_ring = (
            SharedMemoryRingBuffer(
                f"{prefix}_vr",
                dtype=VR_FRAME_DTYPE,
                maxlen=cfg.vr_ring_maxlen,
                create=True,
            )
            if cfg.vr
            else None
        )
        storage.pointcloud_ring = (
            SharedMemoryRingBuffer(
                f"{prefix}_pointcloud",
                dtype=make_pointcloud_frame_dtype(cfg.pointcloud_num_points),
                maxlen=cfg.pointcloud_ring_maxlen,
                create=True,
            )
            if cfg.pointcloud
            else None
        )

        storage.is_running = ctx.Value("b", True, lock=False)

        storage.vr_ready = ctx.Event()
        storage.camera_ready = ctx.Event()
        storage.pointcloud_ready = ctx.Event()

        storage.camera_depth_scale = ctx.Value("d", 0.0, lock=False)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32, lock=False)
        storage.camera_geometry = ctx.Array("c", b"\x00" * 2048, lock=False)

    def close(self) -> bool:
        """Close/unlink every allocated ring; ignore missing segments and report failures."""
        if self._closed:
            return True

        errors: list[str] = []

        def _attempt(operation: str, callback: Any, *, missing_ok: bool = False) -> None:
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    errors.append(operation)
                    logger.warning("SensorChannels close: %s failed", operation, exc_info=True)
            except Exception:
                errors.append(operation)
                logger.warning("SensorChannels close: %s failed", operation, exc_info=True)

        for ring_name in _RING_RESOURCE_NAMES:
            ring = getattr(self, ring_name, None)
            if ring is None:
                continue
            _attempt(f"{ring_name}.close", ring.close)
            _attempt(f"{ring_name}.unlink", ring.unlink, missing_ok=True)

        self._closed = not errors
        if self._closed:
            logger.debug("SensorChannels closed cleanly")
        else:
            logger.error("SensorChannels close incomplete: %s", ", ".join(errors))
        return self._closed

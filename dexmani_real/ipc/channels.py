"""Shared-memory rings, queues, events, and flags for runtime processes."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    ROBOT_COMMAND_DTYPE,
    VR_FRAME_DTYPE,
    make_pointcloud_frame_dtype,
    make_record_sample_dtype,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Wire value of runtime.safety.SafetyState.DISARMED.
DISARMED_SAFETY_STATE_WIRE_VALUE = 0


@dataclass
class RuntimeChannelsConfig:
    """Ring capacities, queue sizes, and camera resolution defaults."""

    camera_ring_maxlen: int = CameraParams.ring_maxlen
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    record_sample_ring_maxlen: int = 16
    pointcloud_num_points: int = 1024
    pointcloud_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: CameraParams().rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: CameraParams().depth_shape)

    arm_home_q_maxsize: int = 2

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.arm_state_ring_maxlen,
            self.hand_state_ring_maxlen,
            self.record_sample_ring_maxlen,
            self.pointcloud_ring_maxlen,
            self.arm_home_q_maxsize,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("RuntimeChannels ring/queue capacities must be positive")
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
    "arm_state_ring",
    "hand_state_ring",
    "robot_command_ring",
    "record_sample_ring",
    "pointcloud_ring",
)
_QUEUE_RESOURCE_NAMES = ("arm_home_q", "arm_home_result_q", "hand_home_q", "hand_home_result_q")
_RECORDER_QUEUE_RESOURCE_NAMES = ("record_control_q", "record_result_q")


@dataclass
class RuntimeChannels:
    """Runtime channels created in Main before spawning child processes."""

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SharedMemoryRingBuffer  # vr -> policy
    arm_state_ring: SharedMemoryRingBuffer  # arm -> policy
    hand_state_ring: SharedMemoryRingBuffer  # hand -> policy
    robot_command_ring: SharedMemoryRingBuffer  # latest current-run target mailbox
    record_sample_ring: SharedMemoryRingBuffer  # policy -> RecorderIO fixed payload
    pointcloud_ring: SharedMemoryRingBuffer  # pointcloud worker -> policy

    arm_home_q: mp.Queue  # requester -> arm HOME (waypoints, final_qpos, run_id, expires_ns)
    arm_home_result_q: Any
    hand_home_q: Any
    hand_home_result_q: Any
    record_control_q: mp.Queue  # policy -> RecorderIO episode boundaries
    record_result_q: mp.Queue  # RecorderIO -> RecorderClient (sole consumer)
    run_id: Any  # controller advances it to invalidate old policy proposals
    # Latest software RUNNING termination; written under motion_lock.
    run_ended_reason: Any

    is_running: Any  # Main -> all
    # Sticky runtime/safety fault: supervision takes the FAULT shutdown path.
    error_state: Any
    estop_request: Any  # sticky emergency-stop request
    quit_requested: Any  # policy -> Main
    start_request: Any  # Main -> policy runner: B (start a new policy run)
    # Main/operator -> policy runner: true only after Main completed the
    # authorized hand-home + collision-checked arm-home sequence.
    physical_home_completed: Any
    # Main/operator -> policy runner: S request.
    stop_request: Any

    safety_state: Any  # SafetyState enum (0-3), Main + policy write
    # Serializes the motion permit and coupled-command ring writer. It is never
    # held across hardware SDK calls.
    motion_lock: Any

    arm_ready: Any
    hand_ready: Any
    policy_ready: Any
    recorder_ready: Any
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
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
        mp_context: Any | None = None,
    ) -> "RuntimeChannels":
        """Create all rings, queues, flags, and events.

        Call once from Main before spawning child processes.
        """
        cfg = config or RuntimeChannelsConfig()
        ctx = mp_context or mp.get_context("spawn")

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)
        storage._closed = False
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx, _rgb_shape, _depth_shape)
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
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int],
    ) -> None:
        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            maxlen=cfg.camera_ring_maxlen,
            create=True,
        )
        storage.vr_ring = SharedMemoryRingBuffer(
            f"{prefix}_vr",
            dtype=VR_FRAME_DTYPE,
            maxlen=cfg.vr_ring_maxlen,
            create=True,
        )
        storage.arm_state_ring = SharedMemoryRingBuffer(
            f"{prefix}_arm_state",
            dtype=ARM_STATE_DTYPE,
            maxlen=cfg.arm_state_ring_maxlen,
            create=True,
        )
        storage.hand_state_ring = SharedMemoryRingBuffer(
            f"{prefix}_hand_state",
            dtype=HAND_STATE_DTYPE,
            maxlen=cfg.hand_state_ring_maxlen,
            create=True,
        )
        storage.robot_command_ring = SharedMemoryRingBuffer(
            f"{prefix}_robot_command",
            dtype=ROBOT_COMMAND_DTYPE,
            maxlen=1,
            create=True,
        )
        storage.record_sample_ring = SharedMemoryRingBuffer(
            f"{prefix}_record_sample",
            dtype=make_record_sample_dtype(rgb_shape, depth_shape),
            maxlen=cfg.record_sample_ring_maxlen,
            create=True,
        )
        storage.pointcloud_ring = SharedMemoryRingBuffer(
            f"{prefix}_pointcloud",
            dtype=make_pointcloud_frame_dtype(cfg.pointcloud_num_points),
            maxlen=cfg.pointcloud_ring_maxlen,
            create=True,
        )

        storage.arm_home_q = ctx.Queue(maxsize=cfg.arm_home_q_maxsize)
        storage.arm_home_result_q = ctx.Queue(maxsize=2)
        storage.hand_home_q = ctx.Queue(maxsize=2)
        storage.hand_home_result_q = ctx.Queue(maxsize=2)
        storage.record_control_q = ctx.Queue(maxsize=8)
        storage.record_result_q = ctx.Queue(maxsize=8)
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

        storage.arm_ready = ctx.Event()
        storage.hand_ready = ctx.Event()
        storage.policy_ready = ctx.Event()
        storage.recorder_ready = ctx.Event()
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

        for queue_name in _QUEUE_RESOURCE_NAMES:
            queue = getattr(self, queue_name, None)
            if queue is None:
                continue
            if _attempt(f"{queue_name}.close", queue.close):
                _attempt(f"{queue_name}.join_thread", queue.join_thread)

        for queue_name in _RECORDER_QUEUE_RESOURCE_NAMES:
            queue = getattr(self, queue_name, None)
            if queue is None:
                continue
            _attempt(f"{queue_name}.cancel_join_thread", queue.cancel_join_thread)
            _attempt(f"{queue_name}.close", queue.close)

        self._closed = not errors
        if self._closed:
            logger.debug("RuntimeChannels closed cleanly")
        else:
            logger.error("RuntimeChannels close incomplete: %s", ", ".join(errors))
        return self._closed


def read_arm_state(shared: RuntimeChannels) -> np.ndarray | None:
    """Read latest arm state from ring. Returns raw structured array or None."""
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_arm_state_dict(shared):
    state = read_arm_state(shared)
    return None if state is None else {name: state[name][0].copy() for name in state.dtype.names}

"""RuntimeChannels — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import camera
from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    ROBOT_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
    VR_FRAME_DTYPE,
    make_pointcloud_frame_dtype,
    make_record_sample_dtype,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ``runtime.safety.SafetyState`` owns the enum; IPC carries this stable wire value
# without importing the runtime state machine back into the data plane.
DISARMED_SAFETY_STATE_WIRE_VALUE = 0


@dataclass
class RuntimeChannelsConfig:
    """Centralized configuration for RuntimeChannels variable sizes and camera defaults.

    Variable ring ``maxlen`` values and camera resolution defaults are gathered
    here so they have a single source of truth rather than being scattered across
    entry points. Fixed wire details remain module constants.

    Usage::

        cfg = RuntimeChannelsConfig()
        shared = RuntimeChannels.create(config=cfg)
    """

    camera_ring_maxlen: int = field(default_factory=lambda: camera.ring_maxlen)
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    record_sample_ring_maxlen: int = 4
    pointcloud_num_points: int = 1024
    pointcloud_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(
        default_factory=lambda: camera.rgb_shape
    )
    camera_depth_shape: tuple[int, int] = field(
        default_factory=lambda: camera.depth_shape
    )

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
            raise ValueError(
                "RuntimeChannels pointcloud_num_points must be a positive integer"
            )

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        pointcloud_num_points: int = 1024,
    ) -> "RuntimeChannelsConfig":
        cam = getattr(runtime, "camera")
        return cls(
            camera_ring_maxlen=int(cam.ring_maxlen),
            camera_rgb_shape=(int(cam.height), int(cam.width), 3),
            camera_depth_shape=(int(cam.height), int(cam.width)),
            pointcloud_num_points=pointcloud_num_points,
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
    """Central data plane — all cross-process state in one place.

    Created by Main before spawning child processes. Each process receives a
    reference and reads/writes its designated rings/queues/flags.
    """

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SharedMemoryRingBuffer  # vr -> policy
    arm_state_ring: SharedMemoryRingBuffer  # arm -> policy
    hand_state_ring: SharedMemoryRingBuffer  # hand -> policy
    robot_command_ring: (
        SharedMemoryRingBuffer  # latest current-run target mailbox
    )
    record_sample_ring: SharedMemoryRingBuffer  # policy -> RecorderIO fixed payload
    pointcloud_ring: SharedMemoryRingBuffer  # pointcloud worker -> policy

    arm_home_q: mp.Queue  # requester -> arm HOME (waypoints, final_qpos, run_id, expires_ns)
    arm_home_result_q: Any
    hand_home_q: Any
    hand_home_result_q: Any
    record_control_q: mp.Queue  # policy -> RecorderIO episode boundaries
    record_result_q: mp.Queue  # RecorderIO -> RecorderClient (sole consumer)
    run_id: Any  # controller advances it to invalidate old policy proposals
    run_started_monotonic_ns: Any  # start of the current RUNNING observation epoch
    # Latest software RUNNING terminal snapshot; safety owns writes under motion_lock.
    run_ended_reason: Any
    recorder_finish_deadline_ns: Any  # recorder-owned; zero while idle
    recorder_completed_ns: Any  # close completed before finish deadline; reset on START
    workflow_failed: Any  # sticky; never reuse IPC after failure/death
    recorder_consumed_sequence: Any
    record_episode_path: Any  # RecorderIO-owned identity for orphan diagnostics

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    # Sticky fail-closed runtime/safety fault latch. When set, supervision
    # takes the FAULT path. The owning workflow decides which failures require
    # this disposition.
    error_state: Any
    # Sticky experiment/session failure latch. When set without a physical/runtime
    # fault, the owning workflow may use verified non-FAULT shutdown while still
    # reporting the session as failed.
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main
    start_request: Any  # Main -> policy runner: B (start a new policy run)
    # Main/operator -> policy runner: true only after Main completed the
    # authorized hand-home + collision-checked arm-home sequence.
    physical_home_completed: Any
    # Main/operator -> policy runner: explicit S request.
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

    camera_depth_scale: Any  # depth scale (mm to meters)
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
                logger.critical(
                    "RuntimeChannels allocation rollback raised", exc_info=True
                )
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
        storage.run_started_monotonic_ns = ctx.Value("Q", 0)
        storage.run_ended_reason = ctx.Value("i", 0)
        # Recorder may be killed at any instruction. Its shared status accesses
        # must not acquire mutexes also needed by surviving workers/monitor.
        # Ready events and consumed sequence have one writer; camera
        # metadata is immutable after camera readiness. These status scalars
        # need no compound transaction. Motion state keeps its existing locks.
        storage.recorder_finish_deadline_ns = ctx.Value("q", 0, lock=False)
        storage.workflow_failed = ctx.Value("b", False, lock=False)
        storage.recorder_completed_ns = ctx.Value("q", 0, lock=False)
        storage.recorder_consumed_sequence = ctx.Value("Q", 0, lock=False)
        storage.record_episode_path = ctx.Array("c", b"\x00" * 4096, lock=False)

        storage.is_running = ctx.Value("b", True, lock=False)
        storage.is_recording = ctx.Value("b", False)
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
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        Cleanup is best-effort across every resource. Underlying close calls are
        idempotent and an already-unlinked shared-memory segment raises
        ``FileNotFoundError``, so a retry can simply repeat the full sequence.

        Returns:
            Whether every owned resource was closed and unlinked successfully.
        """
        if bool(getattr(self, "_closed", False)):
            return True

        errors: list[str] = []

        def _attempt(
            operation: str, callback: Any, *, missing_ok: bool = False
        ) -> bool:
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    errors.append(operation)
                    logger.warning(
                        "RuntimeChannels close: %s failed", operation, exc_info=True
                    )
                    return False
            except Exception:
                errors.append(operation)
                logger.warning(
                    "RuntimeChannels close: %s failed", operation, exc_info=True
                )
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

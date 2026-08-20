"""SharedStorage — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, camera, hand, policy
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.camera_ring import CameraRingBuffer
from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_STATE_DTYPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    POLICY_PLAN_DTYPE,
    RECORD_CONTROL_DTYPE,
    RECORD_STATUS_DTYPE,
    VR_FRAME_DTYPE,
    make_record_sample_dtype,
)

logger = get_logger(__name__)


# SharedStorageConfig — centralized tuning constants


@dataclass
class SharedStorageConfig:
    """Centralized configuration for SharedStorage ring sizes, camera defaults,
    and workspace bounds.

    All ring ``maxlen`` values, camera resolution defaults, and workspace
    boundary constants are gathered here so they have a single source of truth
    rather than being scattered across entry points.

    Usage::

        cfg = SharedStorageConfig()
        shared = SharedStorage.create(config=cfg)
    """

    camera_ring_maxlen: int = field(default_factory=lambda: camera.ring_maxlen)
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    hand_tactile_ring_maxlen: int = 8
    hand_cmd_ring_maxlen: int = 8
    record_control_ring_maxlen: int = 1
    record_sample_ring_maxlen: int = 4
    record_status_ring_maxlen: int = 1
    policy_plan_ring_maxlen: int = 3

    control_hz: float = field(default_factory=lambda: policy.control_hz)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)
    hand_loop_hz: float = field(default_factory=lambda: hand.loop_hz)
    hand_home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            float(value) for value in np.deg2rad(hand.home_qpos_deg)
        )
    )

    camera_rgb_shape: tuple[int, int, int] = field(
        default_factory=lambda: camera.rgb_shape
    )
    camera_depth_shape: tuple[int, int] = field(
        default_factory=lambda: camera.depth_shape
    )

    arm_cmd_ring_maxlen: int = 4
    arm_home_q_maxsize: int = 2

    workspace_bounds: "np.ndarray" = field(
        default_factory=lambda: policy.workspace.as_array()
    )

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.arm_state_ring_maxlen,
            self.hand_state_ring_maxlen,
            self.hand_tactile_ring_maxlen,
            self.hand_cmd_ring_maxlen,
            self.record_control_ring_maxlen,
            self.record_sample_ring_maxlen,
            self.record_status_ring_maxlen,
            self.policy_plan_ring_maxlen,
            self.arm_cmd_ring_maxlen,
            self.arm_home_q_maxsize,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("SharedStorage ring/queue capacities must be positive")
        if min(self.control_hz, self.arm_loop_hz, self.hand_loop_hz) <= 0:
            raise ValueError("SharedStorage action rates must be positive")
        bounds = np.asarray(self.workspace_bounds, dtype=np.float64)
        if (
            bounds.shape != (3, 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError(
                "SharedStorage workspace_bounds must be finite shape (3, 2)"
            )
        hand_home = np.asarray(self.hand_home_qpos_rad, dtype=np.float64)
        if hand_home.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand_home)):
            raise ValueError(
                "SharedStorage hand_home_qpos_rad must be finite shape (12,)"
            )

    @classmethod
    def from_runtime(cls, runtime: object) -> "SharedStorageConfig":
        cam = getattr(runtime, "camera")
        pol = getattr(runtime, "policy")
        arm_cfg = getattr(runtime, "arm")
        hand_cfg = getattr(runtime, "hand")
        bounds = np.array(
            [
                [pol.workspace.x_min, pol.workspace.x_max],
                [pol.workspace.y_min, pol.workspace.y_max],
                [pol.workspace.z_min, pol.workspace.z_max],
            ],
            dtype=np.float64,
        )
        return cls(
            camera_ring_maxlen=int(cam.ring_maxlen),
            camera_rgb_shape=(int(cam.height), int(cam.width), 3),
            camera_depth_shape=(int(cam.height), int(cam.width)),
            control_hz=float(pol.control_hz),
            arm_loop_hz=float(arm_cfg.loop_hz),
            hand_loop_hz=float(hand_cfg.loop_hz),
            hand_home_qpos_rad=tuple(
                float(value) for value in np.deg2rad(hand_cfg.home_qpos_deg)
            ),
            workspace_bounds=bounds,
        )


_RING_RESOURCE_NAMES = (
    "camera_ring",
    "vr_ring",
    "arm_state_ring",
    "hand_state_ring",
    "hand_tactile_ring",
    "hand_cmd_ring",
    "record_control_ring",
    "record_sample_ring",
    "record_status_ring",
    "policy_plan_ring",
    "arm_cmd_ring",
)
_QUEUE_RESOURCE_NAMES = ("arm_home_q",)
_ALLOCATION_ROLLBACK_ATTEMPTS = 2

# Ordered heartbeat slots — one fixed array. Index order is stable across
# processes.
HEARTBEAT_FIELDS: tuple[str, ...] = (
    "arm",
    "hand",
    "policy",
    "recorder",
    "vr",
    "camera",
    "inference",
)
HEARTBEAT_INDEX: dict[str, int] = {name: index for index, name in enumerate(HEARTBEAT_FIELDS)}

# Ordered readiness slots — one fixed array. Per-element access on
# ``ctx.Array`` is atomic, so each flag is a simple 0/1 store; the index order
# is stable across processes.
READY_FIELDS: tuple[str, ...] = (
    "arm",
    "hand",
    "camera",
    "vr",
    "policy",
    "recorder",
    "inference",
)
READY_INDEX: dict[str, int] = {name: index for index, name in enumerate(READY_FIELDS)}

# Poll granularity for wait_ready(). Readiness is a one-shot startup wait with
# generous timeouts; a short poll keeps the observed return value identical to
# Event.wait(timeout) without burning CPU.
_READY_POLL_INTERVAL_S = 0.01


def new_frame(dtype: np.dtype) -> np.ndarray:
    """Allocate a zero-initialized 1-element structured array for ring writes."""
    return np.zeros(1, dtype=dtype)


@dataclass
class SharedStorage:
    """Central data plane — all cross-process state in one place.

    Created by Main before spawning child processes. Each process receives a
    reference and reads/writes its designated rings/queues/flags.
    """

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SharedMemoryRingBuffer  # vr -> policy
    arm_state_ring: SharedMemoryRingBuffer  # arm -> policy
    hand_state_ring: SharedMemoryRingBuffer  # hand -> policy
    hand_tactile_ring: SharedMemoryRingBuffer  # hand -> policy (sparse)
    hand_cmd_ring: SharedMemoryRingBuffer  # policy -> hand
    record_control_ring: SharedMemoryRingBuffer  # policy -> RecorderIO episode boundary
    record_sample_ring: SharedMemoryRingBuffer  # policy -> RecorderIO fixed payload
    record_status_ring: SharedMemoryRingBuffer  # RecorderIO -> policy/main
    policy_plan_ring: SharedMemoryRingBuffer  # inference -> coordinator, latest-wins

    arm_cmd_ring: SharedMemoryRingBuffer  # policy -> arm servo endpoints, latest-wins
    arm_home_q: mp.Queue  # requester -> arm HOME (waypoints, final_qpos, generation)
    arm_command_seq: (
        Any  # all actuator-action producers -> globally unique monotonic IDs
    )
    arm_armed_at_seq: Any  # command seq at arm time; older endpoints are stale
    run_generation: Any  # controller advances it to invalidate old policy proposals
    recorder_consumed_sequence: Any
    action_control_hz: float
    action_lead_time_s: float
    hand_home_qpos_rad: tuple[float, ...]

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    error_state: Any  # arm/hand -> all (sticky latch)
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main

    safety_state: Any  # SafetyState enum (0-3), Main + policy write

    heartbeats: Any  # fixed-order array of per-subsystem heartbeat timestamps (s)

    ready_flags: Any  # fixed-order array of per-subsystem readiness flags (0/1)

    arm_device_identity: Any  # worker-reported canonical identity JSON
    hand_device_identity: Any  # worker-reported canonical identity JSON
    camera_depth_scale: Any  # depth scale (mm to meters)
    camera_K: Any  # 3x3 intrinsics, row-major
    camera_serial: Any  # serial number string
    camera_firmware: Any  # firmware version string
    camera_sdk_version: Any  # pyrealsense2/librealsense version string
    camera_profile: Any  # actual color/depth profile JSON
    camera_pointcloud_config: Any  # resolved pointcloud filter config JSON
    _closed: bool = field(init=False, repr=False, default=False)
    _close_completed_operations: set[str] = field(
        init=False, repr=False, default_factory=set
    )

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: SharedStorageConfig | None = None,
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
        mp_context: Any | None = None,
    ) -> "SharedStorage":
        """Create all rings, queues, flags, events, and heartbeats.

        Call once from Main before spawning child processes.
        """
        cfg = config or SharedStorageConfig()
        ctx = mp_context or mp.get_context("spawn")

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)
        storage._closed = False
        storage._close_completed_operations = set()
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx, _rgb_shape, _depth_shape)
        except BaseException as allocation_error:
            cleanup_succeeded = False
            for _ in range(_ALLOCATION_ROLLBACK_ATTEMPTS):
                try:
                    cleanup_succeeded = storage.close()
                except BaseException:
                    logger.critical(
                        "SharedStorage allocation rollback raised", exc_info=True
                    )
                    raise RuntimeError(
                        "SharedStorage allocation failed and rollback raised"
                    ) from allocation_error
                if cleanup_succeeded:
                    break
            if not cleanup_succeeded:
                raise RuntimeError(
                    "SharedStorage allocation failed and rollback was incomplete"
                ) from allocation_error
            raise

        logger.info("SharedStorage created (prefix=%s)", prefix)
        return storage

    @staticmethod
    def _allocate_resources(
        storage: "SharedStorage",
        prefix: str,
        cfg: SharedStorageConfig,
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
        storage.vr_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_vr",
            dtype=vr_frame_dtype(),
            maxlen=cfg.vr_ring_maxlen,
        )
        storage.arm_state_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_arm_state",
            dtype=ARM_STATE_DTYPE,
            maxlen=cfg.arm_state_ring_maxlen,
        )
        storage.hand_state_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_hand_state",
            dtype=HAND_STATE_DTYPE,
            maxlen=cfg.hand_state_ring_maxlen,
        )
        storage.hand_tactile_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_hand_tactile",
            dtype=HAND_TACTILE_DTYPE,
            maxlen=cfg.hand_tactile_ring_maxlen,
        )
        storage.hand_cmd_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_hand_cmd",
            dtype=HAND_COMMAND_DTYPE,
            maxlen=cfg.hand_cmd_ring_maxlen,
        )
        storage.record_control_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_record_control",
            dtype=RECORD_CONTROL_DTYPE,
            maxlen=cfg.record_control_ring_maxlen,
        )
        storage.record_sample_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_record_sample",
            dtype=make_record_sample_dtype(rgb_shape, depth_shape),
            maxlen=cfg.record_sample_ring_maxlen,
        )
        storage.record_status_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_record_status",
            dtype=RECORD_STATUS_DTYPE,
            maxlen=cfg.record_status_ring_maxlen,
        )
        storage.policy_plan_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_policy_plan",
            dtype=POLICY_PLAN_DTYPE,
            maxlen=cfg.policy_plan_ring_maxlen,
        )

        storage.arm_cmd_ring = SharedMemoryRingBuffer.create_or_replace(
            f"{prefix}_arm_cmd",
            dtype=ARM_COMMAND_DTYPE,
            maxlen=cfg.arm_cmd_ring_maxlen,
        )
        storage.arm_home_q = ctx.Queue(maxsize=cfg.arm_home_q_maxsize)
        storage.arm_command_seq = ctx.Value("Q", 0)
        # Sequence captured when motion was armed; the worker ignores endpoints
        # created before it so a re-arm never replays pre-disarm commands.
        storage.arm_armed_at_seq = ctx.Value("Q", 0)
        storage.run_generation = ctx.Value("Q", 1)
        storage.recorder_consumed_sequence = ctx.Value("Q", 0)
        storage.action_control_hz = float(cfg.control_hz)
        storage.action_lead_time_s = 2.0 / min(
            float(cfg.arm_loop_hz), float(cfg.hand_loop_hz)
        )
        storage.hand_home_qpos_rad = tuple(
            float(value) for value in cfg.hand_home_qpos_rad
        )

        storage.is_running = ctx.Value("b", True)
        storage.is_recording = ctx.Value("b", False)
        storage.error_state = ctx.Value("b", False)
        storage.estop_request = ctx.Value("b", False)
        storage.quit_requested = ctx.Value("b", False)

        storage.safety_state = ctx.Value("i", int(SafetyState.DISARMED))

        storage.heartbeats = ctx.Array("d", [0.0] * len(HEARTBEAT_FIELDS))

        storage.ready_flags = ctx.Array("b", len(READY_FIELDS))

        storage.arm_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.hand_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.camera_depth_scale = ctx.Value("d", 0.0)
        storage.camera_K = ctx.Array("d", [0.0] * 9)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32)
        storage.camera_firmware = ctx.Array("c", b"\x00" * 64)
        storage.camera_sdk_version = ctx.Array("c", b"\x00" * 64)
        storage.camera_profile = ctx.Array("c", b"\x00" * 2048)
        storage.camera_pointcloud_config = ctx.Array("c", b"\x00" * 2048)

    def close(self) -> bool:
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        Cleanup is best-effort across every resource.  A failed operation is
        retryable, while operations that already succeeded are not repeated.

        Returns:
            Whether every owned resource was closed and unlinked successfully.
        """
        if bool(getattr(self, "_closed", False)):
            return True

        completed: set[str] = getattr(self, "_close_completed_operations", set())
        if not isinstance(completed, set):
            completed = set()
        self._close_completed_operations = completed
        expected: set[str] = set()
        _close_errors: list[str] = []

        def _attempt(
            operation: str, callback: Any, *, missing_ok: bool = False
        ) -> bool:
            expected.add(operation)
            if operation in completed:
                return True
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    _close_errors.append(operation)
                    logger.warning(
                        "SharedStorage close: %s failed", operation, exc_info=True
                    )
                    return False
            except Exception:
                _close_errors.append(operation)
                logger.warning(
                    "SharedStorage close: %s failed", operation, exc_info=True
                )
                return False
            completed.add(operation)
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
            queue_close = f"{queue_name}.close"
            queue_join = f"{queue_name}.join_thread"
            expected.add(queue_join)
            if _attempt(queue_close, queue.close):
                _attempt(queue_join, queue.join_thread)

        self._closed = not _close_errors and expected.issubset(completed)
        if self._closed:
            logger.info("SharedStorage closed cleanly")
        else:
            logger.error("SharedStorage close incomplete: %s", ", ".join(_close_errors))
        return self._closed

    def set_heartbeat(self, name: str, value_s: float) -> None:
        """Record a fresh heartbeat timestamp (s) for *name* (a HEARTBEAT_FIELDS key)."""
        self.heartbeats[HEARTBEAT_INDEX[name]] = float(value_s)

    def get_heartbeat(self, name: str) -> float:
        """Return the last recorded heartbeat timestamp (s) for *name*, or 0.0."""
        return float(self.heartbeats[HEARTBEAT_INDEX[name]])

    def set_ready(self, name: str) -> None:
        """Mark *name* (a READY_FIELDS key) ready."""
        self.ready_flags[READY_INDEX[name]] = 1

    def is_ready(self, name: str) -> bool:
        """Return True when *name* is ready."""
        return bool(self.ready_flags[READY_INDEX[name]])

    def wait_ready(self, name: str, timeout: float) -> bool:
        """Block until *name* is ready or *timeout* seconds elapse; True if ready.

        Equivalent to ``Event.wait(timeout)`` in its return value. Readiness is a
        one-shot startup wait, so a short poll replaces the kernel-level event
        wait without any observable latency difference.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_ready(name):
                return True
            time.sleep(_READY_POLL_INTERVAL_S)
        return self.is_ready(name)


def vr_frame_dtype() -> np.dtype:
    """VR frame dtype — mirrors vr_receiver_process frame output."""
    return VR_FRAME_DTYPE


# Shared ring read/write helpers


def read_arm_state(shared: "SharedStorage") -> "np.ndarray | None":
    """Read latest arm state from ring. Returns raw structured array or None."""
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_hand_state(shared: "SharedStorage") -> "np.ndarray | None":
    """Read latest hand state from ring. Returns raw structured array or None."""
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_arm_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest arm state from ring. Return dict of numpy arrays or None.

    Fields: qpos(7), qvel(7), tau(7), error_code, connected, tracking_err,
            last_cmd_seq, last_cmd_is_hold, source/publish timestamps,
            state_valid.
    Callers must validate fields they depend on (e.g. ``np.all(np.isfinite(d["qpos"]))``).
    The EEF pose is not published; derive it from ``qpos`` via
    ``planning.kinematics.make_arm_fk()`` when needed.
    """
    data = read_arm_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "qvel": np.asarray(data["qvel"][0], dtype=np.float64),
        "tau": np.asarray(data["tau"][0], dtype=np.float64),
        "error_code": int(data["error_code"][0]),
        "connected": bool(data["connected"][0]),
        "tracking_err": float(data["tracking_err"][0]),
        "last_cmd_seq": int(data["last_cmd_seq"][0]),
        "last_cmd_is_hold": bool(data["last_cmd_is_hold"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
    }


def read_hand_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest hand state from ring. Return dict of numpy arrays or None.

    Fields include qpos/current/tactile data, tactile validity, hardware/read
    health, and the last hand action ID accepted by the worker/SDK.
    """
    data = read_hand_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "current": np.asarray(data["current"][0], dtype=np.float64),
        "tactile_sum": np.asarray(data["tactile_sum"][0], dtype=np.float64),
        "tactile_sum_valid": bool(data["tactile_sum_valid"][0]),
        "tactile_contact": np.asarray(data["tactile_contact"][0], dtype=bool),
        "connected": bool(data["connected"][0]),
        "error_state": bool(data["error_state"][0]),
        "qpos_stale": bool(data["qpos_stale"][0]),
        "last_cmd_seq": int(data["last_cmd_seq"][0]),
        "last_cmd_qpos": np.asarray(data["last_cmd_qpos"][0], dtype=np.float64),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
        "send_healthy": bool(data["send_healthy"][0]),
        "read_healthy": bool(data["read_healthy"][0]),
    }

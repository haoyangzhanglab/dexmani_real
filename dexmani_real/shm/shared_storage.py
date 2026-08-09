"""SharedStorage — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from queue import Empty, Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, camera, hand, policy
from dexmani_real.policy.action_protocol import ACK_DTYPE, ARM_COMMAND_DTYPE, COMMIT_DTYPE, HAND_COMMAND_DTYPE
from dexmani_real.policy.inference_process import INFERENCE_CANDIDATE_DTYPE
from dexmani_real.recording.io_process import RECORD_CONTROL_DTYPE, RECORD_STATUS_DTYPE, make_record_sample_dtype
from dexmani_real.robot.safety import SafetyState
from dexmani_real.runtime.status import ComponentPhase, ExitReason, FaultCode
from dexmani_real.shm.ring_buffer import CameraRingBuffer
from dexmani_real.shm.robot_ring import SeqlockRingBuffer
from dexmani_real.utils.log import get_logger

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
    record_sample_ring_maxlen: int = 4

    control_hz: float = field(default_factory=lambda: policy.control_hz)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)
    hand_loop_hz: float = field(default_factory=lambda: hand.loop_hz)
    hand_home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.deg2rad(hand.home_qpos_deg))
    )

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: camera.rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: camera.depth_shape)
    camera_pc_shape: tuple[int, int] = field(default_factory=lambda: camera.pointcloud_shape)

    arm_action_q_maxsize: int = 2

    workspace_bounds: "np.ndarray" = field(default_factory=lambda: policy.workspace.as_array())

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.arm_state_ring_maxlen,
            self.hand_state_ring_maxlen,
            self.hand_tactile_ring_maxlen,
            self.hand_cmd_ring_maxlen,
            self.record_sample_ring_maxlen,
            self.arm_action_q_maxsize,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("SharedStorage ring/queue capacities must be positive")
        if min(self.control_hz, self.arm_loop_hz, self.hand_loop_hz) <= 0:
            raise ValueError("SharedStorage action rates must be positive")
        bounds = np.asarray(self.workspace_bounds, dtype=np.float64)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("SharedStorage workspace_bounds must be finite shape (3, 2)")
        hand_home = np.asarray(self.hand_home_qpos_rad, dtype=np.float64)
        if hand_home.shape != (12,) or not np.all(np.isfinite(hand_home)):
            raise ValueError("SharedStorage hand_home_qpos_rad must be finite shape (12,)")

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
            camera_pc_shape=(int(cam.pointcloud_num_points), 6),
            control_hz=float(pol.control_hz),
            arm_loop_hz=float(arm_cfg.loop_hz),
            hand_loop_hz=float(hand_cfg.loop_hz),
            hand_home_qpos_rad=tuple(float(value) for value in np.deg2rad(hand_cfg.home_qpos_deg)),
            workspace_bounds=bounds,
        )


HOME_SENTINEL = "__HOME__"


@dataclass(frozen=True)
class HomeRequest:
    """Densely validated, sparse-milestone homing command for ``arm_loop``."""

    request_id: int
    waypoints: "np.ndarray"
    final_qpos: "np.ndarray"
    execution_timeout_s: float


@dataclass(frozen=True)
class HomeResult:
    """Completion acknowledgement produced by the arm worker for one home request."""

    request_id: int
    success: bool
    reason: str
    final_qpos: "np.ndarray"
    completed_at_s: float


def _describe_band_diff(wrapped: "np.ndarray", canonical: "np.ndarray") -> str:
    """Describe which equivalent joints differ between wrapped and canonical home.

    Returns a short string like ``"J7:-360→0°"`` or ``"same band"``.
    """
    import numpy as np

    delta_deg = np.rad2deg(np.abs(wrapped - canonical))
    _EQ_JOINT_NAMES = {0: "J1", 2: "J3", 4: "J5", 6: "J7"}
    parts: list[str] = []
    for _ji, _name in _EQ_JOINT_NAMES.items():
        if delta_deg[_ji] > 1.0:
            parts.append(f"{_name}:{np.rad2deg(wrapped[_ji]):.0f}→{np.rad2deg(canonical[_ji]):.0f}°")
    return ", ".join(parts) if parts else "same band"


ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (7,)),
        ("qvel", "<f8", (7,)),
        ("tau", "<f8", (7,)),
        ("eef_pos", "<f8", (3,)),
        ("eef_rot6d", "<f8", (6,)),
        ("error_code", "<i4"),
        ("connected", "<u1"),
        ("mode", "<i4"),
        ("tracking_err", "<f8"),
        # Last arm action successfully accepted by the SDK. Monotonic timestamps
        # share one host clock across processes and make queue/SDK latency measurable.
        ("last_cmd_seq", "<u8"),
        ("last_cmd_created_s", "<f8"),
        ("last_cmd_received_s", "<f8"),
        ("last_cmd_applied_s", "<f8"),
        ("last_cmd_queue_latency_s", "<f8"),
        ("last_cmd_apply_latency_s", "<f8"),
        ("last_cmd_sdk_duration_s", "<f8"),
        ("last_cmd_is_hold", "<u1"),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
        ("timestamp", "<f8"),
    ]
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (12,)),
        ("current", "<f8", (12,)),
        ("tactile_sum", "<f8", (5, 3)),
        ("tactile_contact", "<u1", (5,)),
        ("error_state", "<u1"),
        ("connected", "<u1"),
        ("qpos_stale", "<u1"),
        ("commboard_err", "<i4", (12,)),
        ("jointboard_err", "<i4", (12,)),
        ("tipboard_err", "<i4", (12,)),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
        ("send_healthy", "<u1"),
        ("read_healthy", "<u1"),
        ("timestamp", "<f8"),
    ]
)  # no tactile_force — that is carried by hand_tactile_ring

HAND_CMD_DTYPE = HAND_COMMAND_DTYPE

COMPONENT_STATUS_DTYPE = np.dtype(
    [
        ("component", "S24"),
        ("phase", "<u1"),
        ("fault_code", "<u2"),
        ("exit_reason", "<u1"),
        ("generation", "<u8"),
        ("updated_monotonic_ns", "<u8"),
        ("detail", "S160"),
    ],
    align=True,
)

HAND_TACTILE_DTYPE = np.dtype(
    [
        ("tactile_force", "<f8", (5, 120, 3)),
        ("source_monotonic_ns", "<u8"),
        ("fresh", "<u1"),
        ("calibrated", "<u1"),
        ("unit_code", "<u1"),  # 0=unknown, 1=newton (vendor conversion provenance required)
    ]
)


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
    vr_ring: SeqlockRingBuffer  # vr -> policy
    arm_state_ring: SeqlockRingBuffer  # arm -> policy
    hand_state_ring: SeqlockRingBuffer  # hand -> policy
    hand_tactile_ring: SeqlockRingBuffer  # hand -> policy (sparse)
    hand_cmd_ring: SeqlockRingBuffer  # policy -> hand
    action_commit_ring: SeqlockRingBuffer  # coordinator -> arm + hand
    arm_ack_ring: SeqlockRingBuffer  # arm -> coordinator
    hand_ack_ring: SeqlockRingBuffer  # hand -> coordinator
    component_status_ring: SeqlockRingBuffer  # workers -> supervisor diagnostics
    record_control_ring: SeqlockRingBuffer  # policy -> RecorderIO episode boundary
    record_sample_ring: SeqlockRingBuffer  # policy -> RecorderIO fixed payload
    record_status_ring: SeqlockRingBuffer  # RecorderIO -> policy/main
    inference_candidate_ring: SeqlockRingBuffer  # Inference -> coordinator only

    arm_action_q: mp.Queue  # policy -> arm, maxsize=2
    arm_home_result_q: mp.Queue  # arm -> requester; request_id correlates replies
    arm_command_seq: Any  # all arm-action producers -> globally unique monotonic IDs
    session_generation: Any
    policy_epoch: Any
    recorder_consumed_sequence: Any
    action_control_hz: float
    action_lead_time_s: float
    action_validity_s: float
    hand_home_qpos_rad: tuple[float, ...]

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    error_state: Any  # arm/hand -> all (sticky latch)
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main

    safety_state: Any  # SafetyState enum (0-3), Main + policy write

    arm_heartbeat_s: Any
    hand_heartbeat_s: Any
    policy_heartbeat_s: Any
    recorder_heartbeat_s: Any
    inference_heartbeat_s: Any
    vr_heartbeat_s: Any
    camera_heartbeat_s: Any

    arm_ready: Any  # -> Main
    hand_ready: Any  # -> Main
    camera_ready: Any  # -> Main
    vr_ready: Any  # -> Main
    policy_ready: Any  # -> Main, only after policy/backend warmup
    inference_ready: Any  # optional capability -> Main
    recorder_ready: Any  # optional capability -> Main
    component_status_lock: Any  # serializes the multi-producer diagnostic ring

    arm_device_identity: Any  # worker-reported canonical identity JSON
    hand_device_identity: Any  # worker-reported canonical identity JSON
    camera_depth_scale: Any  # depth scale (mm to meters)
    camera_K: Any  # 3x3 intrinsics, row-major
    camera_serial: Any  # serial number string
    camera_firmware: Any  # firmware version string
    camera_sdk_version: Any  # pyrealsense2/librealsense version string
    camera_profile: Any  # actual color/depth profile JSON
    camera_observation_required: Any  # learned policy requests camera payload publication

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

        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=_rgb_shape,
            depth_shape=_depth_shape,
            pc_shape=cfg.camera_pc_shape,
            maxlen=cfg.camera_ring_maxlen,
            create=True,
        )
        storage.vr_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_vr",
            dtype=vr_frame_dtype(),
            maxlen=cfg.vr_ring_maxlen,
        )
        storage.arm_state_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_arm_state",
            dtype=ARM_STATE_DTYPE,
            maxlen=cfg.arm_state_ring_maxlen,
        )
        storage.hand_state_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_state",
            dtype=HAND_STATE_DTYPE,
            maxlen=cfg.hand_state_ring_maxlen,
        )
        storage.hand_tactile_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_tactile",
            dtype=HAND_TACTILE_DTYPE,
            maxlen=cfg.hand_tactile_ring_maxlen,
        )
        storage.hand_cmd_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_cmd",
            dtype=HAND_CMD_DTYPE,
            maxlen=cfg.hand_cmd_ring_maxlen,
        )
        storage.action_commit_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_action_commit", dtype=COMMIT_DTYPE, maxlen=8
        )
        storage.arm_ack_ring = SeqlockRingBuffer.create_or_replace(f"{prefix}_arm_ack", dtype=ACK_DTYPE, maxlen=16)
        storage.hand_ack_ring = SeqlockRingBuffer.create_or_replace(f"{prefix}_hand_ack", dtype=ACK_DTYPE, maxlen=16)
        storage.component_status_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_component_status", dtype=COMPONENT_STATUS_DTYPE, maxlen=32
        )
        storage.record_control_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_control", dtype=RECORD_CONTROL_DTYPE, maxlen=8
        )
        storage.record_sample_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_sample",
            dtype=make_record_sample_dtype(_rgb_shape, _depth_shape, cfg.camera_pc_shape),
            maxlen=cfg.record_sample_ring_maxlen,
        )
        storage.record_status_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_status", dtype=RECORD_STATUS_DTYPE, maxlen=16
        )
        storage.inference_candidate_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_inference_candidate", dtype=INFERENCE_CANDIDATE_DTYPE, maxlen=16
        )

        storage.arm_action_q = ctx.Queue(maxsize=cfg.arm_action_q_maxsize)
        storage.arm_home_result_q = ctx.Queue(maxsize=cfg.arm_action_q_maxsize)
        storage.arm_command_seq = ctx.Value("Q", 0)
        storage.session_generation = ctx.Value("Q", time.monotonic_ns())
        storage.policy_epoch = ctx.Value("Q", 1)
        storage.recorder_consumed_sequence = ctx.Value("Q", 0)
        storage.action_control_hz = float(cfg.control_hz)
        storage.action_lead_time_s = 2.0 / min(float(cfg.arm_loop_hz), float(cfg.hand_loop_hz))
        storage.action_validity_s = 1.0 / float(cfg.control_hz)
        storage.hand_home_qpos_rad = tuple(float(value) for value in cfg.hand_home_qpos_rad)

        storage.is_running = ctx.Value("b", True)
        storage.is_recording = ctx.Value("b", False)
        storage.error_state = ctx.Value("b", False)
        storage.estop_request = ctx.Value("b", False)
        storage.quit_requested = ctx.Value("b", False)

        storage.safety_state = ctx.Value("i", int(SafetyState.DISARMED))

        storage.arm_heartbeat_s = ctx.Value("d", 0.0)
        storage.hand_heartbeat_s = ctx.Value("d", 0.0)
        storage.policy_heartbeat_s = ctx.Value("d", 0.0)
        storage.recorder_heartbeat_s = ctx.Value("d", 0.0)
        storage.inference_heartbeat_s = ctx.Value("d", 0.0)
        storage.vr_heartbeat_s = ctx.Value("d", 0.0)
        storage.camera_heartbeat_s = ctx.Value("d", 0.0)

        storage.arm_ready = ctx.Event()
        storage.hand_ready = ctx.Event()
        storage.camera_ready = ctx.Event()
        storage.vr_ready = ctx.Event()
        storage.policy_ready = ctx.Event()
        storage.inference_ready = ctx.Event()
        storage.recorder_ready = ctx.Event()
        storage.component_status_lock = ctx.Lock()

        storage.arm_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.hand_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.camera_depth_scale = ctx.Value("d", 0.0)
        storage.camera_K = ctx.Array("d", [0.0] * 9)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32)
        storage.camera_firmware = ctx.Array("c", b"\x00" * 64)
        storage.camera_sdk_version = ctx.Array("c", b"\x00" * 64)
        storage.camera_profile = ctx.Array("c", b"\x00" * 2048)
        storage.camera_observation_required = ctx.Value("b", False)

        logger.info("SharedStorage created (prefix=%s)", prefix)
        storage._closed = False
        return storage

    def close(self) -> None:
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        """
        if bool(getattr(self, "_closed", False)):
            return
        self._closed = True
        _close_errors: list[str] = []

        for ring_name, ring in (
            ("camera_ring", self.camera_ring),
            ("vr_ring", self.vr_ring),
            ("arm_state_ring", self.arm_state_ring),
            ("hand_state_ring", self.hand_state_ring),
            ("hand_tactile_ring", self.hand_tactile_ring),
            ("hand_cmd_ring", self.hand_cmd_ring),
            ("action_commit_ring", self.action_commit_ring),
            ("arm_ack_ring", self.arm_ack_ring),
            ("hand_ack_ring", self.hand_ack_ring),
            ("component_status_ring", self.component_status_ring),
            ("record_control_ring", self.record_control_ring),
            ("record_sample_ring", self.record_sample_ring),
            ("record_status_ring", self.record_status_ring),
            ("inference_candidate_ring", self.inference_candidate_ring),
        ):
            try:
                ring.close()  # type: ignore[attr-defined]
            except Exception:
                _close_errors.append(f"{ring_name}.close() failed")

            try:
                ring.unlink()  # type: ignore[attr-defined]
            except FileNotFoundError:
                pass  # already unlinked by another process — expected
            except Exception:
                _close_errors.append(f"{ring_name}.unlink() failed")

        for queue_name, queue in (
            ("arm_action_q", self.arm_action_q),
            ("arm_home_result_q", self.arm_home_result_q),
        ):
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                _close_errors.append(f"{queue_name} cleanup failed")

        if _close_errors:
            logger.warning("SharedStorage close: %d error(s): %s", len(_close_errors), "; ".join(_close_errors))
        else:
            logger.info("SharedStorage closed cleanly")


def vr_frame_dtype() -> np.dtype:
    """VR frame dtype — mirrors vr_receiver_process frame output."""
    return np.dtype(
        [
            ("wrist_pos", "<f8", (3,)),
            ("wrist_quat_wxyz", "<f8", (4,)),
            ("landmarks", "<f8", (21, 3)),
            ("head_pos", "<f8", (3,)),
            ("head_quat_wxyz", "<f8", (4,)),
            ("recv_ts_ns", "<u8"),
            ("source_ts_ns", "<u8"),
            ("sequence_id", "<u8"),
            ("source_frame_seq", "<u8"),
            ("local_recv_ns", "<u8"),
            ("side", "<i4"),
        ],
        align=True,
    )


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


def write_hand_cmd(shared: "SharedStorage", qpos: "np.ndarray", *, safety_gate: "Any | None" = None) -> bool:
    """Publish a hand target together with a measured arm hold."""
    arm_state = read_arm_state(shared)
    if arm_state is None:
        return False
    arm_qpos = np.asarray(arm_state["qpos"][0], dtype=np.float64)
    from dexmani_real.policy.action_protocol import publish_joint_targets

    return (
        publish_joint_targets(
            shared,
            arm_qpos,
            np.asarray(qpos, dtype=np.float64),
            is_hold=True,
            safety_gate=safety_gate,
        )
        is not None
    )


def publish_component_status(
    shared: Any,
    component: str,
    phase: ComponentPhase,
    *,
    fault_code: FaultCode = FaultCode.NONE,
    exit_reason: ExitReason = ExitReason.NONE,
    detail: str = "",
) -> None:
    """Publish one structured component health transition when available.

    Minimal offline mocks and older external façades may omit the diagnostic
    ring; status reporting is observability-only and must not mask the worker's
    primary safety behavior in that compatibility case.
    """
    ring = getattr(shared, "component_status_ring", None)
    if ring is None:
        return
    frame = new_frame(COMPONENT_STATUS_DTYPE)
    frame["component"][0] = component.encode("utf-8")[:24]
    frame["phase"][0] = int(phase)
    frame["fault_code"][0] = int(fault_code)
    frame["exit_reason"][0] = int(exit_reason)
    generation = getattr(shared, "session_generation", None)
    frame["generation"][0] = int(generation.value) if generation is not None else 0
    frame["updated_monotonic_ns"][0] = time.monotonic_ns()
    frame["detail"][0] = detail.encode("utf-8")[:160]
    lock = getattr(shared, "component_status_lock", None)
    if lock is None:
        ring.write(frame)
    else:
        with lock:
            ring.write(frame)


def hand_home_converge(
    shared: SharedStorage,
    home_qpos: np.ndarray,
    *,
    timeout_s: float = 5.0,
    tol_deg: float = 5.0,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    safety_gate: "Any | None" = None,
) -> "tuple[bool, np.ndarray | None]":
    """Poll hand_state_ring until hand converges to home_qpos.

    Returns ``(reached, final_qpos_or_none)``.
    """
    tol = np.deg2rad(tol_deg)
    deadline = time.monotonic() + timeout_s
    requested_after_s = time.monotonic()
    first = True

    while time.monotonic() < deadline:
        if check_is_running and not shared.is_running.value:
            break
        if heartbeat:
            shared.policy_heartbeat_s.value = time.monotonic()
        if not write_hand_cmd(shared, home_qpos, safety_gate=safety_gate):
            if verbose:
                print("  hand: coordinated home command was rejected", flush=True)
            return False, None
        hs = read_hand_state(shared)
        if hs is not None:
            current = np.asarray(hs["qpos"][0], dtype=np.float64)
            fresh = float(hs["timestamp"][0]) >= requested_after_s
            healthy = bool(hs["connected"][0]) and not bool(hs["error_state"][0])
            if fresh and healthy and np.all(np.isfinite(current)):
                err = float(np.max(np.abs(current - home_qpos)))
                if err < tol:
                    if verbose:
                        print("  hand: home reached", flush=True)
                    return True, current.copy()
                if verbose and first:
                    print(f"  hand: homing... (max_err={np.rad2deg(err):.0f}°)", flush=True)
                    first = False
        # Allow the two-worker lead time plus one actuator tick before
        # replacing the next latest-wins hand endpoint.
        time.sleep(0.1)

    if verbose:
        print(f"  hand: home settle timeout after {timeout_s:.0f}s — proceeding", flush=True)
    return False, None


def read_arm_state_k(shared: "SharedStorage", k: int) -> "list[np.ndarray]":
    """Read up to *k* most recent arm state frames (oldest-first), may be shorter than *k*."""
    frames = shared.arm_state_ring.get_last_k(k)
    return [data for data, _ts, _seq in frames]


def read_hand_state_k(shared: "SharedStorage", k: int) -> "list[np.ndarray]":
    """Read up to *k* most recent hand state frames (oldest-first), may be shorter than *k*."""
    frames = shared.hand_state_ring.get_last_k(k)
    return [data for data, _ts, _seq in frames]


# ═══════════════════════════════════════════════ Shared entry-point helpers


def read_arm_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest arm state from ring. Return dict of numpy arrays or None.

    Fields: qpos(7), qvel(7), tau(7), eef_pos(3), eef_rot6d(6),
            error_code, connected, tracking_err, and last-command timing.
    Callers must validate fields they depend on (e.g. ``np.all(np.isfinite(d["qpos"]))``).
    """
    data = read_arm_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "qvel": np.asarray(data["qvel"][0], dtype=np.float64),
        "tau": np.asarray(data["tau"][0], dtype=np.float64),
        "eef_pos": np.asarray(data["eef_pos"][0], dtype=np.float64),
        "eef_rot6d": np.asarray(data["eef_rot6d"][0], dtype=np.float64),
        "error_code": int(data["error_code"][0]),
        "connected": bool(data["connected"][0]),
        "tracking_err": float(data["tracking_err"][0]),
        "last_cmd_seq": int(data["last_cmd_seq"][0]),
        "last_cmd_created_s": float(data["last_cmd_created_s"][0]),
        "last_cmd_received_s": float(data["last_cmd_received_s"][0]),
        "last_cmd_applied_s": float(data["last_cmd_applied_s"][0]),
        "last_cmd_queue_latency_s": float(data["last_cmd_queue_latency_s"][0]),
        "last_cmd_apply_latency_s": float(data["last_cmd_apply_latency_s"][0]),
        "last_cmd_sdk_duration_s": float(data["last_cmd_sdk_duration_s"][0]),
        "last_cmd_is_hold": bool(data["last_cmd_is_hold"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
    }


def read_hand_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest hand state from ring. Return dict of numpy arrays or None.

    Fields: qpos(12), current(12), tactile_sum(5,3), tactile_contact(5),
            connected, error_state, qpos_stale.
    """
    data = read_hand_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "current": np.asarray(data["current"][0], dtype=np.float64),
        "tactile_sum": np.asarray(data["tactile_sum"][0], dtype=np.float64),
        "tactile_contact": np.asarray(data["tactile_contact"][0], dtype=bool),
        "connected": bool(data["connected"][0]),
        "error_state": bool(data["error_state"][0]),
        "qpos_stale": bool(data["qpos_stale"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
        "send_healthy": bool(data["send_healthy"][0]),
        "read_healthy": bool(data["read_healthy"][0]),
    }


def shutdown_processes(
    shared: "SharedStorage",
    procs: "list[Any]",
    *,
    graceful_timeout_s: float = 5.0,
) -> None:
    """Compatibility wrapper for verified graceful→terminate→kill shutdown."""
    from dexmani_real.runtime.processes import shutdown_processes_verified

    report = shutdown_processes_verified(shared, procs, graceful_timeout_s=graceful_timeout_s)
    if report.exits:
        print("  shutdown: " + "  ".join(f"{item.name}={item.escalation}:{item.exitcode}" for item in report.exits))


def wait_for_arm_home(
    shared: "SharedStorage",
    home_qpos: "np.ndarray",
    *,
    request_id: int | None = None,
    requested_after_s: float | None = None,
    timeout_s: float = 20.0,
    tol_rad: float = 0.03,
    heartbeat: bool = False,
    verbose: bool = True,
) -> bool:
    """Wait for a fresh, correlated arm-worker homing acknowledgement.

    New callers pass *request_id*.  The legacy state-only path is retained for
    compatibility with external callers, but requires a frame newer than
    *requested_after_s* and a healthy, connected arm.  Both paths fail early on
    shutdown, sticky error, FAULT, stale arm heartbeat, or controller error.
    """
    _deadline = time.monotonic() + timeout_s
    _not_before = time.monotonic() if requested_after_s is None else float(requested_after_s)
    _abort_reason: str | None = None
    _fault_ack_deadline: float | None = None
    while time.monotonic() < _deadline:
        _now = time.monotonic()
        if heartbeat:
            shared.policy_heartbeat_s.value = _now

        _result = None
        if request_id is not None:
            try:
                _result = shared.arm_home_result_q.get(timeout=min(0.1, max(0.0, _deadline - _now)))
            except Empty:
                pass
            if _result is not None and (not isinstance(_result, HomeResult) or _result.request_id != request_id):
                logger.warning("wait_for_arm_home: discarded stale/malformed result %r", _result)
                continue
        if isinstance(_result, HomeResult):
            _q = np.asarray(_result.final_qpos, dtype=np.float64)
            _converged = (
                _q.shape == home_qpos.shape
                and np.all(np.isfinite(_q))
                and float(np.max(np.abs(_q - home_qpos))) < tol_rad
            )
            if _result.success and _converged:
                if verbose:
                    print("  arm: home reached", flush=True)
                return True
            if verbose:
                print(f"  arm: home failed — {_result.reason}", flush=True)
            return False

        if not shared.is_running.value:
            _abort_reason = "shutdown requested"
            break
        _fault_reason: str | None = None
        if shared.error_state.value:
            _fault_reason = "sticky error_state set"
        elif shared.safety_state.value == int(SafetyState.FAULT):
            _fault_reason = "safety state is FAULT"
        if _fault_reason is not None:
            # multiprocessing.Queue.put() may return before its feeder thread
            # makes the correlated HomeResult visible.  The arm worker latches
            # error_state immediately after publishing a failed result, so give
            # that acknowledgement one bounded grace window before reporting a
            # generic fault.  This preserves the precise SDK/timeout reason.
            if request_id is not None:
                if _fault_ack_deadline is None:
                    _fault_ack_deadline = min(_deadline, _now + 0.25)
                if _now < _fault_ack_deadline:
                    continue
            _abort_reason = _fault_reason
            break
        _arm_hb = float(shared.arm_heartbeat_s.value)
        if _arm_hb > 0.0 and _now - _arm_hb > 1.0:
            _abort_reason = f"arm heartbeat stale ({_now - _arm_hb:.1f}s)"
            break
        if request_id is not None:
            continue

        _as = read_arm_state(shared)
        if _as is not None:
            _q = np.asarray(_as["qpos"][0], dtype=np.float64)
            _fresh = float(_as["timestamp"][0]) >= _not_before
            _healthy = bool(_as["connected"][0]) and int(_as["error_code"][0]) == 0
            if _fresh and _healthy and np.all(np.isfinite(_q)):
                if float(np.max(np.abs(_q - home_qpos))) < tol_rad:
                    if verbose:
                        print("  arm: home reached", flush=True)
                    return True
        time.sleep(0.1)
    if verbose:
        if _abort_reason is not None:
            print(f"  arm: home wait aborted — {_abort_reason}", flush=True)
        else:
            print(f"  arm: home acknowledgement timed out after {timeout_s:.1f}s", flush=True)
    return False


def _estimate_home_timeout_s(waypoints: "np.ndarray") -> float:
    """Deadline derived from milestone path length and feedback settle overhead."""
    if len(waypoints) < 2:
        return 10.0
    segment_motion = np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)
    nominal_s = float(np.sum(segment_motion)) / max(np.deg2rad(arm.homing.max_speed_deg_s), 1e-6)
    moving_segments = int(np.count_nonzero(segment_motion > 1e-9))
    settle_s = moving_segments * arm.homing.target_timeout_s
    return max(10.0, 2.0 * nominal_s + settle_s + 5.0)


def _format_home_candidate_rejection(candidate: "dict[str, Any]") -> str:
    """Format one path-candidate diagnostic without dumping large arrays."""
    name = str(candidate.get("name", "unknown"))
    reason = str(candidate.get("reason", "unknown"))
    if reason == "self_collision":
        collision = candidate.get("collision") or {}
        pairs = collision.get("collision_pairs", []) if isinstance(collision, dict) else []
        pair_names = [f"{pair.get('link1', '?')}<->{pair.get('link2', '?')}" for pair in pairs[:2]]
        pair_text = ",".join(pair_names) if pair_names else "pair unavailable"
        return f"{name}: self_collision sample={candidate.get('collision_waypoint_index', '?')} ({pair_text})"
    if reason == "table_clearance":
        clearance_mm = 1000.0 * float(candidate.get("clearance_m", float("nan")))
        return (
            f"{name}: table_clearance sample={candidate.get('table_waypoint_index', '?')} "
            f"margin={clearance_mm:+.1f}mm"
        )
    if reason == "workspace":
        return f"{name}: workspace segment={candidate.get('workspace_segment_index', '?')}"
    detail = str(candidate.get("detail", "")).strip()
    return f"{name}: {reason}" + (f" ({detail})" if detail else "")


def send_arm_home(
    shared: "SharedStorage",
    home_qpos: "np.ndarray",
    *,
    planner: "Any | None" = None,
    table_z_surface_m: float = 0.0,
    current_qpos: "np.ndarray | None" = None,
    queue_timeout: float = 2.0,
    converge_timeout_s: float = 15.0,
    heartbeat: bool = True,
    verbose: bool = True,
) -> bool:
    """Send arm to home via collision-safe path and wait for convergence.

    Encapsulates the full home sequence used by all entry points:
    1. Read current qpos (from *current_qpos* or ``arm_state_ring``).
    2. Densely validate collision-safe segments and retain sparse milestones.
    3. Queue a correlated ``HomeRequest`` to ``arm_action_q``.
    4. Wait for convergence via ``wait_for_arm_home``.

    A planner is required. Missing state, planning errors, and unsafe paths
    fail closed and are never converted into direct interpolation.

    Returns True if home reached, False on timeout or error.
    """
    from dexmani_real.planning.path_utils import plan_band_alignment_path, plan_joint_home_path

    # HOME is a motion command, not a fault-recovery command.  arm_loop stops
    # consuming actions after a sticky fault, so rejecting here prevents
    # impossible requests from filling the bounded action queue.
    if not shared.is_running.value:
        if verbose:
            print("  arm: homing cancelled — shutdown in progress", flush=True)
        return False
    if shared.error_state.value or shared.safety_state.value == int(SafetyState.FAULT):
        if verbose:
            print("  arm: homing cancelled — system is in FAULT; restart after inspection", flush=True)
        return False
    if shared.safety_state.value not in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
        if verbose:
            print("  arm: homing cancelled — safety state is not ARMED/RUNNING", flush=True)
        return False

    # ── Step 1: resolve current qpos ──
    if current_qpos is None:
        _as = read_arm_state(shared)
        _state_age_s = float("inf") if _as is None else time.monotonic() - float(_as["timestamp"][0])
        if (
            _as is None
            or _state_age_s > 0.5
            or not bool(_as["connected"][0])
            or int(_as["error_code"][0]) != 0
            or not np.all(np.isfinite(_as["qpos"][0]))
        ):
            if verbose:
                print(
                    f"  arm: current state is stale/unhealthy (age={_state_age_s:.2f}s) — homing cancelled", flush=True
                )
            return False
        current_qpos = np.asarray(_as["qpos"][0], dtype=np.float64)
    else:
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        if current_qpos.shape != (7,) or not np.all(np.isfinite(current_qpos)):
            if verbose:
                print("  arm: invalid current qpos — homing cancelled", flush=True)
            return False

    if planner is None:
        if verbose:
            print("  arm: no collision planner — homing cancelled", flush=True)
        return False

    # ── Step 2: plan collision-safe path to wrapped home ──
    _home_path_report: dict[str, Any] = {}
    try:
        _waypoints = plan_joint_home_path(
            current_qpos,
            home_qpos,
            planner,
            table_z_surface_m=table_z_surface_m,
            report=_home_path_report,
        )
    except Exception as exc:
        logger.warning("send_arm_home: planning failed", exc_info=True)
        if verbose:
            print(f"  arm: home path planning failed — holding ({exc})", flush=True)
        return False

    if _waypoints is not None and len(_waypoints) == 0:
        if verbose:
            _candidate_text = "; ".join(
                _format_home_candidate_rejection(candidate)
                for candidate in _home_path_report.get("candidates", [])
                if not candidate.get("safe", False)
            )
            _qpos_text = np.array2string(np.rad2deg(current_qpos), precision=1, separator=",")
            print(
                "  arm: no validated home-path candidate — holding\n"
                f"       current_qpos_deg={_qpos_text}\n"
                f"       rejected={_candidate_text or 'no candidate diagnostics'}",
                flush=True,
            )
        return False

    if verbose and _waypoints is not None:
        _selected = _home_path_report.get("selected_candidate", "unknown")
        print(f"  arm: home path selected={_selected} milestones={len(_waypoints)}", flush=True)

    _wrapped_home = (
        _waypoints[-1].copy()
        if _waypoints is not None and len(_waypoints) > 0
        else planner.ik_mgr.nearest_equivalent_qpos(home_qpos, current_qpos)
    )
    if _waypoints is None:
        _waypoints = np.empty((0, 7), dtype=np.float64)

    # ── Step 2b: plan band-alignment path (wrapped_home → canonical_home) ──
    # plan_joint_home_path wraps home_qpos to the nearest 2π band of the
    # arm's current encoder position.  The returned waypoints end at this
    # wrapped position (_home).  For strategy learning we need the arm at
    # the canonical home_qpos, so we append a collision-checked alignment
    # segment that rotates only the band-mismatched equivalent joints.
    try:
        _align_path = plan_band_alignment_path(_wrapped_home, home_qpos, planner, table_z_surface_m=table_z_surface_m)
    except Exception as exc:
        logger.warning("send_arm_home: band-alignment planning failed", exc_info=True)
        if verbose:
            print(f"  arm: band-alignment planning failed — holding ({exc})", flush=True)
        return False

    if _align_path is not None:
        if len(_align_path) == 0:
            if verbose:
                _desc = _describe_band_diff(_wrapped_home, home_qpos)
                print(f"  arm: band-alignment UNSAFE ({_desc}) — holding", flush=True)
            return False
        # Keep the first alignment waypoint only when there is no main path.
        _tail = _align_path[1:] if len(_waypoints) > 0 else _align_path
        _waypoints = np.concatenate([_waypoints, _tail], axis=0)
        if verbose:
            _desc = _describe_band_diff(_wrapped_home, home_qpos)
            print(f"  arm: band-alignment appended ({len(_tail)} milestones, {_desc})", flush=True)

    # ── Step 3: queue HOME_SENTINEL ──
    _request_id = time.monotonic_ns()
    _execution_timeout_s = max(float(converge_timeout_s), _estimate_home_timeout_s(_waypoints))
    # A prior caller may have abandoned a result.  Homing is serialized, so it
    # is safe to drain stale acknowledgements before publishing the new ID.
    while True:
        try:
            shared.arm_home_result_q.get_nowait()
        except Empty:
            break
    try:
        _request = HomeRequest(
            request_id=_request_id,
            waypoints=np.asarray(_waypoints, dtype=np.float64),
            final_qpos=np.asarray(home_qpos, dtype=np.float64).copy(),
            execution_timeout_s=_execution_timeout_s,
        )
        _requested_at_s = time.monotonic()
        shared.arm_action_q.put((HOME_SENTINEL, _request), timeout=queue_timeout)
    except Full:
        if verbose:
            print("  arm: action queue is full — homing request was not queued", flush=True)
        return False
    except Exception:
        logger.warning("send_arm_home: failed to queue HOME request", exc_info=True)
        if verbose:
            print("  arm: failed to queue homing request", flush=True)
        return False

    # ── Step 4: wait for convergence ──
    return wait_for_arm_home(
        shared,
        home_qpos,
        request_id=_request_id,
        requested_after_s=_requested_at_s,
        timeout_s=_execution_timeout_s + 2.0,
        tol_rad=np.deg2rad(2.0),
        heartbeat=heartbeat,
        verbose=verbose,
    )


def run_supervisor(
    shared: "SharedStorage",
    procs: "list",
    proc_names: "list[str]",
    heartbeat_fields: "dict[str, Any]",
    *,
    status_interval_s: float = 30.0,
    heartbeat_timeouts_s: "dict[str, float] | None" = None,
    supervisor_hz: float | None = None,
) -> "tuple[str, bool]":
    """Run the standard supervisor loop with resolved heartbeat settings.

    Returns ``(exit_reason, normal_exit)``.  *exit_reason* describes why the
    supervisor stopped; *normal_exit* is True for user-requested clean exits
    (Q key or KeyboardInterrupt), False for faults.

    The caller should have already transitioned to ARMED before calling this
    and must handle shutdown + DISARMED transition after it returns.
    """
    import time as _time

    from dexmani_real.config.defaults import safety
    from dexmani_real.robot.safety import SafetyState, transition
    from dexmani_real.runtime.processes import supervisor_exit_reason
    from dexmani_real.runtime.status import ExitReason

    _start_time = _time.monotonic()
    _last_status_s = _start_time
    _exit_reason = "unknown"
    normal_exit = False
    _supervisor_hz = float(safety.supervisor_hz if supervisor_hz is None else supervisor_hz)
    if _supervisor_hz <= 0:
        raise ValueError("supervisor_hz must be positive")
    configured_timeouts = safety.heartbeat_timeouts if heartbeat_timeouts_s is None else heartbeat_timeouts_s
    _timeouts = {name: float(configured_timeouts[name]) for name in heartbeat_fields}
    if any(timeout <= 0 for timeout in _timeouts.values()):
        raise ValueError("heartbeat timeouts must be positive")

    try:
        while True:
            _now = _time.monotonic()
            _heartbeat_ages = {
                name: (_now - float(value.value) if float(value.value) > 0 else float("inf"))
                for name, value in heartbeat_fields.items()
            }
            _reason = supervisor_exit_reason(shared, procs, _heartbeat_ages, _timeouts)
            if _reason is ExitReason.ESTOP:
                _exit_reason = "e-stop requested"
                transition(shared, SafetyState.FAULT)
                break
            if _reason is ExitReason.STICKY_FAULT:
                _exit_reason = "error_state set"
                transition(shared, SafetyState.FAULT)
                break
            if _reason is ExitReason.WORKER_DEATH:
                _dead_names = [process.name for process in procs if process.exitcode is not None]
                _exit_reason = f"process died: {_dead_names}"
                transition(shared, SafetyState.FAULT)
                break
            if _reason is ExitReason.HEARTBEAT_TIMEOUT:
                _stale = [name for name, age in _heartbeat_ages.items() if age > _timeouts[name]]
                _exit_reason = f"heartbeat timeout: {_stale}"
                transition(shared, SafetyState.FAULT)
                break
            if _reason is ExitReason.EXPLICIT_QUIT:
                normal_exit = True
                _exit_reason = "explicit quit"
                break

            # 4. Periodic status print.
            if _now - _last_status_s >= status_interval_s:
                _runtime_m = (_now - _start_time) / 60.0
                _safety = shared.safety_state.value
                _hb_ages = ", ".join(f"{n}={_now - float(heartbeat_fields[n].value):.1f}s" for n in proc_names)
                print(
                    f"  [supervisor]  runtime={_runtime_m:.1f}min  safety={_safety}  hb_age=({_hb_ages})",
                    flush=True,
                )
                _last_status_s = _now

            _time.sleep(1.0 / _supervisor_hz)

    except KeyboardInterrupt:
        _exit_reason = "KeyboardInterrupt"
        normal_exit = True
        shared.is_running.value = False

    return _exit_reason, normal_exit


def wait_subsystem_ready(
    shared: "SharedStorage",
    ready_checks: "list[tuple[str, Any, float]]",
    procs: "list[Any]",
) -> bool:
    """Wait for each ``(name, event, timeout_s)`` ready event to be set.

    Checks ``error_state`` and process liveness on every poll tick.
    Returns True if all subsystems are ready, False if any fail.

    The caller is responsible for printing pre-wait user messages
    (e.g. "put on Quest headset") before calling this function.
    """
    for name, ev, timeout in ready_checks:
        _deadline = time.monotonic() + timeout
        _ok = False
        _logged = False
        while time.monotonic() < _deadline:
            if ev.is_set():
                _ok = True
                break
            if shared.error_state.value:
                logger.error("subsystem=%s init failed: error_state set", name)
                _logged = True
                break
            if not all(p.is_alive() for p in procs):
                _dead_names = [p.name for p in procs if not p.is_alive()]
                logger.error(
                    "subsystem=%s init failed: process(es) %s exited prematurely",
                    name,
                    _dead_names,
                )
                _logged = True
                break
            time.sleep(0.2)
        if not _ok and not _logged:
            logger.error("subsystem=%s ready_timeout=%ds", name, timeout)
        if not _ok:
            return False
    return True


def print_health_summary(shared: "SharedStorage") -> None:
    """Print a pre-flight health summary from ring data (arm, hand, VR, camera)."""
    print("\n── Health Check ──")

    # Arm
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is not None:
        arm_data, _, _ = arm_result
        arm_connected = bool(arm_data["connected"][0])
        arm_error = int(arm_data["error_code"][0])
        arm_qpos = np.asarray(arm_data["qpos"][0], dtype=np.float64)
        arm_qpos_ok = int(np.all(np.isfinite(arm_qpos)))
        arm_ok = arm_connected and arm_error == 0 and bool(arm_qpos_ok)
        print(
            f"  arm   {'OK' if arm_ok else 'FAIL':>4s}  connected={int(arm_connected)}  "
            f"error={arm_error}  qpos_ok={arm_qpos_ok}"
        )
    else:
        print("  arm   ----  (no data yet)")

    # Hand
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_data, _, _ = hand_result
        hand_connected = bool(hand_data["connected"][0])
        hand_error = bool(hand_data["error_state"][0])
        hand_qpos_stale = bool(hand_data["qpos_stale"][0])
        hand_qpos = np.asarray(hand_data["qpos"][0], dtype=np.float64)
        hand_qpos_ok = int(np.all(np.isfinite(hand_qpos)))
        hand_ok = hand_connected and not hand_error and bool(hand_qpos_ok)
        stale_note = " stale=1" if hand_qpos_stale else ""
        print(
            f"  hand  {'OK' if hand_ok else 'FAIL':>4s}  connected={int(hand_connected)}  "
            f"error={int(hand_error)}  qpos_ok={hand_qpos_ok}{stale_note}"
        )
    else:
        print("  hand  ----  (no data yet)")

    # VR
    vr_result = shared.vr_ring.read_latest()
    if vr_result is not None:
        vr_data, _, _ = vr_result
        vr_age_s = (
            (time.monotonic_ns() - int(vr_data["local_recv_ns"][0])) / 1e9 if vr_data["local_recv_ns"][0] > 0 else -1
        )
        print(f"  vr     OK   age={vr_age_s:.1f}s  seq={int(vr_data['sequence_id'][0])}")
    else:
        print("  vr    ----  (no data yet)")

    # Camera
    cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
    if cam_serial_bytes:
        print(f"  cam    OK   serial={cam_serial_bytes.decode()}")
    elif shared.camera_heartbeat_s.value > 0:
        print("  cam    OK   serial=unknown")
    else:
        print("  cam   ----  (no data yet)")

    print("──")
    sys.stdout.flush()

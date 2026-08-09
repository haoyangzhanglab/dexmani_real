"""Unified action safety boundary and arm/hand prepare/commit protocol.

Only this module is allowed to access the raw arm action queue, hand command
ring, or commit ring.  Backends, replay, keyboard teleoperation, calibration,
and policy mapping produce :class:`ActionCandidate` values; the safety gate
turns them into validated candidates and this publisher serializes them into
fixed NumPy records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import IntEnum
from queue import Full
from typing import Any, Callable

import numpy as np

from dexmani_real.policy.runtime import ActionCandidate, ActionChunk, ActionSpec, FrozenArrayMap, ObservationSnapshot
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class AckStatus(IntEnum):
    RECEIVED = 1
    PREPARED = 2
    APPLIED = 3
    REJECTED = 4
    SDK_FAILED = 5
    STOPPED = 6


class RejectReason(IntEnum):
    NONE = 0
    INVALID_SHAPE = 1
    NONFINITE = 2
    WRONG_SESSION = 3
    OLD_EPOCH = 4
    OUT_OF_ORDER = 5
    EXPIRED = 6
    NOT_COMMITTED = 7
    JOINT_LIMIT = 8
    SAFETY_STATE = 9
    PREPARE_TIMEOUT = 10
    SDK_ERROR = 11


_COMMAND_COMMON_FIELDS = [
    ("session_generation", "<u8"),
    ("policy_epoch", "<u8"),
    ("observation_id", "<u8"),
    ("action_id", "<u8"),
    ("chunk_id", "<u8"),
    ("step_index", "<u4"),
    ("created_monotonic_ns", "<u8"),
    ("target_monotonic_ns", "<u8"),
    ("valid_until_monotonic_ns", "<u8"),
    ("is_hold", "<u1"),
]

ARM_COMMAND_DTYPE = np.dtype(_COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", (7,))], align=True)
HAND_COMMAND_DTYPE = np.dtype(_COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", (12,))], align=True)
COMMIT_DTYPE = np.dtype(
    [
        ("session_generation", "<u8"),
        ("policy_epoch", "<u8"),
        ("observation_id", "<u8"),
        ("action_id", "<u8"),
        ("chunk_id", "<u8"),
        ("step_index", "<u4"),
        ("created_monotonic_ns", "<u8"),
        ("committed_monotonic_ns", "<u8"),
        ("target_monotonic_ns", "<u8"),
        ("valid_until_monotonic_ns", "<u8"),
        ("is_hold", "<u1"),
    ],
    align=True,
)
ACK_DTYPE = np.dtype(
    [
        ("session_generation", "<u8"),
        ("policy_epoch", "<u8"),
        ("observation_id", "<u8"),
        ("action_id", "<u8"),
        ("chunk_id", "<u8"),
        ("step_index", "<u4"),
        ("status", "<u1"),
        ("reject_reason", "<u2"),
        ("sdk_code", "<i4"),
        ("received_monotonic_ns", "<u8"),
        ("prepared_monotonic_ns", "<u8"),
        ("applied_monotonic_ns", "<u8"),
    ],
    align=True,
)


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    candidate: ActionCandidate | None
    reason: str = ""
    delta_clamped: bool = False


@dataclass(frozen=True)
class ActionSafetyGateConfig:
    arm_joint_lower_rad: tuple[float, ...]
    arm_joint_upper_rad: tuple[float, ...]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]
    arm_max_velocity_rad_s: float
    hand_max_velocity_rad_s: float
    observation_max_age_s: float = 0.25
    require_geometry_checks: bool = True

    def __post_init__(self) -> None:
        arm_lower = np.asarray(self.arm_joint_lower_rad, dtype=np.float64)
        arm_upper = np.asarray(self.arm_joint_upper_rad, dtype=np.float64)
        hand_lower = np.asarray(self.hand_joint_lower_rad, dtype=np.float64)
        hand_upper = np.asarray(self.hand_joint_upper_rad, dtype=np.float64)
        if arm_lower.shape != (7,) or arm_upper.shape != (7,):
            raise ValueError("arm gate limits must have seven entries")
        if hand_lower.shape != (12,) or hand_upper.shape != (12,):
            raise ValueError("hand gate limits must have twelve entries")
        if (
            not np.all(np.isfinite(np.concatenate((arm_lower, arm_upper, hand_lower, hand_upper))))
            or np.any(arm_lower > arm_upper)
            or np.any(hand_lower > hand_upper)
        ):
            raise ValueError("gate joint limits must be finite and ordered")
        rates = (self.arm_max_velocity_rad_s, self.hand_max_velocity_rad_s, self.observation_max_age_s)
        if not all(np.isfinite(value) and value > 0 for value in rates):
            raise ValueError("gate velocity/observation-age limits must be finite and positive")


class ActionSafetyGate:
    """The sole raw-candidate to safe-candidate conversion boundary."""

    def __init__(
        self,
        config: ActionSafetyGateConfig,
        *,
        workspace_check: Callable[[np.ndarray, np.ndarray], bool] | None = None,
        transition_collision_check: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None = None,
        table_clearance_check: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None = None,
    ) -> None:
        self.config = config
        self._workspace_check = workspace_check
        self._transition_collision_check = transition_collision_check
        self._table_clearance_check = table_clearance_check

    def evaluate(
        self,
        candidate: ActionCandidate,
        *,
        snapshot: ObservationSnapshot,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
        expected_session_generation: int,
        expected_policy_epoch: int,
        now_monotonic_ns: int | None = None,
        dt_s: float,
    ) -> GateResult:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if candidate.representation != "joint_position" or candidate.units != "rad" or candidate.frame != "robot_joint":
            return GateResult(False, None, "unsupported action representation/units/frame")
        if candidate.observation_id != snapshot.observation_id:
            return GateResult(False, None, "candidate observation_id does not match snapshot")
        if (
            candidate.session_generation != expected_session_generation
            or snapshot.session_generation != expected_session_generation
        ):
            return GateResult(False, None, "session generation mismatch")
        if candidate.policy_epoch != expected_policy_epoch:
            return GateResult(False, None, "policy epoch mismatch")
        if (
            candidate.valid_until_monotonic_ns < now_ns
            or candidate.target_monotonic_ns > candidate.valid_until_monotonic_ns
        ):
            return GateResult(False, None, "action expired")
        age_ns = now_ns - snapshot.anchor_monotonic_ns
        if age_ns < 0 or age_ns > int(self.config.observation_max_age_s * 1e9):
            return GateResult(False, None, "observation stale or from the future")
        if not candidate.is_hold and any(
            not bool(np.all(snapshot.valid_history_mask[name])) for name in snapshot.valid_history_mask
        ):
            return GateResult(False, None, "observation contains invalid history")
        if not np.isfinite(dt_s) or dt_s <= 0:
            return GateResult(False, None, "invalid action dt")

        arm_start = np.asarray(current_arm_qpos, dtype=np.float64)
        hand_start = np.asarray(current_hand_qpos, dtype=np.float64)
        if arm_start.shape != (7,) or hand_start.shape != (12,):
            return GateResult(False, None, "invalid current joint state shape")
        if not np.all(np.isfinite(arm_start)) or not np.all(np.isfinite(hand_start)):
            return GateResult(False, None, "current joint state contains NaN/Inf")

        arm_end = (
            arm_start.copy() if candidate.arm_qpos is None else np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        )
        hand_end = (
            hand_start.copy()
            if candidate.hand_qpos is None
            else np.asarray(candidate.hand_qpos, dtype=np.float64).copy()
        )
        if (
            arm_end.shape != (7,)
            or hand_end.shape != (12,)
            or not np.all(np.isfinite(arm_end))
            or not np.all(np.isfinite(hand_end))
        ):
            return GateResult(False, None, "invalid candidate joint shape/values")

        arm_low = np.asarray(self.config.arm_joint_lower_rad, dtype=np.float64)
        arm_high = np.asarray(self.config.arm_joint_upper_rad, dtype=np.float64)
        hand_low = np.asarray(self.config.hand_joint_lower_rad, dtype=np.float64)
        hand_high = np.asarray(self.config.hand_joint_upper_rad, dtype=np.float64)
        if np.any(arm_end < arm_low) or np.any(arm_end > arm_high):
            return GateResult(False, None, "arm joint limit violation")
        if np.any(hand_end < hand_low) or np.any(hand_end > hand_high):
            return GateResult(False, None, "hand joint limit violation")

        arm_delta = self.config.arm_max_velocity_rad_s * dt_s
        hand_delta = self.config.hand_max_velocity_rad_s * dt_s
        safe_arm = np.clip(arm_end, arm_start - arm_delta, arm_start + arm_delta)
        safe_hand = np.clip(hand_end, hand_start - hand_delta, hand_start + hand_delta)
        delta_clamped = not np.array_equal(safe_arm, arm_end) or not np.array_equal(safe_hand, hand_end)

        if self.config.require_geometry_checks and (
            self._workspace_check is None
            or self._transition_collision_check is None
            or self._table_clearance_check is None
        ):
            return GateResult(False, None, "geometry safety checks unavailable")
        for name, check, args in (
            ("workspace", self._workspace_check, (arm_start, safe_arm)),
            ("collision", self._transition_collision_check, (arm_start, safe_arm, hand_start, safe_hand)),
            ("table", self._table_clearance_check, (arm_start, safe_arm, hand_start, safe_hand)),
        ):
            if check is None:
                continue
            try:
                if not bool(check(*args)):
                    return GateResult(False, None, f"{name} transition rejected")
            except Exception:
                logger.warning("ActionSafetyGate: %s check failed closed", name, exc_info=True)
                return GateResult(False, None, f"{name} check failed")

        safe_candidate = replace(
            candidate,
            arm_qpos=safe_arm if candidate.arm_qpos is not None else None,
            hand_qpos=safe_hand if candidate.hand_qpos is not None else None,
        )
        return GateResult(True, safe_candidate, delta_clamped=delta_clamped)


def planner_action_safety_gate(
    config: ActionSafetyGateConfig,
    *,
    planner: Any,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    transition_step_rad: float = 0.02,
) -> ActionSafetyGate:
    """Build the canonical geometry-aware gate around a configured planner.

    The table check samples the conservative Cartesian product of arm and hand
    progress because the two workers can apply a committed endpoint up to one
    worker tick apart.  Planner state is restored to the measured hand pose on
    rejection and advanced to the accepted endpoint on success.
    """
    if not np.isfinite(table_z_surface_m) or not np.isfinite(hand_safety_margin_m):
        raise ValueError("table surface and hand safety margin must be finite")
    if transition_step_rad <= 0 or not np.isfinite(transition_step_rad):
        raise ValueError("transition_step_rad must be finite and positive")

    def table_clearance_check(
        arm_start: np.ndarray,
        arm_end: np.ndarray,
        hand_start: np.ndarray,
        hand_end: np.ndarray,
    ) -> bool:
        arm_steps = max(1, int(np.ceil(np.max(np.abs(arm_end - arm_start)) / transition_step_rad)))
        hand_steps = max(1, int(np.ceil(np.max(np.abs(hand_end - hand_start)) / transition_step_rad)))
        try:
            for arm_alpha in np.linspace(0.0, 1.0, arm_steps + 1):
                arm_sample = arm_start + arm_alpha * (arm_end - arm_start)
                for hand_alpha in np.linspace(0.0, 1.0, hand_steps + 1):
                    hand_sample = hand_start + hand_alpha * (hand_end - hand_start)
                    planner.set_hand_qpos(hand_sample)
                    minimum_z = float(planner.collision_model.minimum_hand_frame_z(arm_sample))
                    if not np.isfinite(minimum_z) or minimum_z - hand_safety_margin_m < table_z_surface_m:
                        planner.set_hand_qpos(hand_start)
                        return False
        except Exception:
            planner.set_hand_qpos(hand_start)
            raise
        planner.set_hand_qpos(hand_end)
        return True

    return ActionSafetyGate(
        config,
        workspace_check=planner.is_workspace_segment_safe,
        transition_collision_check=planner.collision_model.check_transition_collision_free,
        table_clearance_check=table_clearance_check,
    )


def make_command_frame(candidate: ActionCandidate, *, actuator: str) -> np.ndarray:
    if actuator == "arm":
        if candidate.arm_qpos is None:
            raise ValueError("candidate has no arm command")
        dtype = ARM_COMMAND_DTYPE
        qpos = candidate.arm_qpos
    elif actuator == "hand":
        if candidate.hand_qpos is None:
            raise ValueError("candidate has no hand command")
        dtype = HAND_COMMAND_DTYPE
        qpos = candidate.hand_qpos
    else:
        raise ValueError(f"unknown actuator {actuator!r}")
    frame = np.zeros(1, dtype=dtype)
    for name in (
        "session_generation",
        "policy_epoch",
        "observation_id",
        "action_id",
        "chunk_id",
        "step_index",
        "created_monotonic_ns",
        "target_monotonic_ns",
        "valid_until_monotonic_ns",
        "is_hold",
    ):
        frame[name][0] = getattr(candidate, name)
    frame["qpos_cmd"][0] = qpos
    return frame


def make_ack(
    command: np.ndarray,
    status: AckStatus,
    *,
    reject_reason: RejectReason = RejectReason.NONE,
    sdk_code: int = 0,
    received_monotonic_ns: int = 0,
    prepared_monotonic_ns: int = 0,
    applied_monotonic_ns: int = 0,
) -> np.ndarray:
    frame = np.zeros(1, dtype=ACK_DTYPE)
    for name in ("session_generation", "policy_epoch", "observation_id", "action_id", "chunk_id", "step_index"):
        frame[name][0] = command[name][0]
    frame["status"][0] = int(status)
    frame["reject_reason"][0] = int(reject_reason)
    frame["sdk_code"][0] = int(sdk_code)
    frame["received_monotonic_ns"][0] = int(received_monotonic_ns)
    frame["prepared_monotonic_ns"][0] = int(prepared_monotonic_ns)
    frame["applied_monotonic_ns"][0] = int(applied_monotonic_ns)
    return frame


def make_stopped_ack(*, applied_monotonic_ns: int | None = None) -> np.ndarray:
    """Build the zero-identity lifecycle ACK emitted after a confirmed stop."""
    frame = np.zeros(1, dtype=ACK_DTYPE)
    frame["status"][0] = int(AckStatus.STOPPED)
    frame["applied_monotonic_ns"][0] = (
        time.monotonic_ns() if applied_monotonic_ns is None else int(applied_monotonic_ns)
    )
    return frame


def command_matches_commit(command: np.ndarray, commit: np.ndarray) -> bool:
    """Return whether a commit exactly authorizes one prepared command."""
    if (
        not isinstance(command, np.ndarray)
        or command.shape != (1,)
        or command.dtype not in (ARM_COMMAND_DTYPE, HAND_COMMAND_DTYPE)
        or not isinstance(commit, np.ndarray)
        or commit.shape != (1,)
        or commit.dtype != COMMIT_DTYPE
    ):
        return False
    for name in (
        "session_generation",
        "policy_epoch",
        "observation_id",
        "action_id",
        "chunk_id",
        "step_index",
        "created_monotonic_ns",
        "target_monotonic_ns",
        "valid_until_monotonic_ns",
        "is_hold",
    ):
        if int(commit[name][0]) != int(command[name][0]):
            return False
    committed_ns = int(commit["committed_monotonic_ns"][0])
    return int(command["created_monotonic_ns"][0]) <= committed_ns <= int(command["target_monotonic_ns"][0])


def validate_worker_command(
    command: np.ndarray,
    *,
    dtype: np.dtype,
    expected_session_generation: int,
    minimum_policy_epoch: int,
    last_action_id: int,
    now_monotonic_ns: int,
    joint_lower_rad: np.ndarray,
    joint_upper_rad: np.ndarray,
) -> RejectReason:
    if not isinstance(command, np.ndarray) or command.shape != (1,) or command.dtype != dtype:
        return RejectReason.INVALID_SHAPE
    qpos = np.asarray(command["qpos_cmd"][0], dtype=np.float64)
    if not np.all(np.isfinite(qpos)):
        return RejectReason.NONFINITE
    if int(command["session_generation"][0]) != expected_session_generation:
        return RejectReason.WRONG_SESSION
    if int(command["policy_epoch"][0]) != minimum_policy_epoch:
        return RejectReason.OLD_EPOCH
    if int(command["action_id"][0]) <= last_action_id:
        return RejectReason.OUT_OF_ORDER
    if (
        int(command["target_monotonic_ns"][0]) < now_monotonic_ns
        or int(command["valid_until_monotonic_ns"][0]) < now_monotonic_ns
    ):
        return RejectReason.EXPIRED
    if np.any(qpos < joint_lower_rad) or np.any(qpos > joint_upper_rad):
        return RejectReason.JOINT_LIMIT
    return RejectReason.NONE


class SafeCommandPublisher:
    """The only owner of raw actuator IPC writes."""

    def __init__(self, shared: Any) -> None:
        self.shared = shared

    def prepare(self, candidate: ActionCandidate, *, timeout_s: float = 0.05) -> None:
        if candidate.arm_qpos is not None:
            arm_frame = make_command_frame(candidate, actuator="arm")
            try:
                self.shared.arm_action_q.put(arm_frame, block=True, timeout=timeout_s)
            except Full as exc:
                raise TimeoutError("arm action queue full") from exc
        if candidate.hand_qpos is not None:
            self.shared.hand_cmd_ring.write(make_command_frame(candidate, actuator="hand"))

    def commit(self, candidate: ActionCandidate) -> None:
        frame = np.zeros(1, dtype=COMMIT_DTYPE)
        frame["session_generation"][0] = candidate.session_generation
        frame["policy_epoch"][0] = candidate.policy_epoch
        frame["observation_id"][0] = candidate.observation_id
        frame["action_id"][0] = candidate.action_id
        frame["chunk_id"][0] = candidate.chunk_id
        frame["step_index"][0] = candidate.step_index
        frame["created_monotonic_ns"][0] = candidate.created_monotonic_ns
        frame["committed_monotonic_ns"][0] = time.monotonic_ns()
        frame["target_monotonic_ns"][0] = candidate.target_monotonic_ns
        frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
        frame["is_hold"][0] = candidate.is_hold
        self.shared.action_commit_ring.write(frame)

    @staticmethod
    def _ack_matches(ack: np.ndarray, candidate: ActionCandidate) -> bool:
        if not isinstance(ack, np.ndarray) or ack.shape != (1,) or ack.dtype != ACK_DTYPE:
            return False
        return all(
            int(ack[name][0]) == int(getattr(candidate, name))
            for name in ("session_generation", "policy_epoch", "observation_id", "action_id", "chunk_id", "step_index")
        )

    @classmethod
    def _prepared(cls, ack_ring: Any, candidate: ActionCandidate) -> bool:
        result = ack_ring.read_latest()
        if result is None:
            return False
        data = result[0]
        return cls._ack_matches(data, candidate) and int(data["status"][0]) == int(AckStatus.PREPARED)

    def publish(self, candidate: ActionCandidate, *, prepare_timeout_s: float = 0.05) -> bool:
        """Prepare both enabled actuators, wait for ACKs, then commit atomically."""
        self.prepare(candidate, timeout_s=prepare_timeout_s)
        deadline = time.monotonic() + prepare_timeout_s
        while time.monotonic() < deadline:
            arm_ready = candidate.arm_qpos is None or self._prepared(self.shared.arm_ack_ring, candidate)
            hand_ready = candidate.hand_qpos is None or self._prepared(self.shared.hand_ack_ring, candidate)
            if arm_ready and hand_ready:
                self.commit(candidate)
                return True
            time.sleep(0.001)
        logger.error("SafeCommandPublisher: prepare timeout for action_id=%d", candidate.action_id)
        return False

    @classmethod
    def _ack_status(cls, ack_ring: Any, candidate: ActionCandidate) -> AckStatus | None:
        result = ack_ring.read_latest()
        if result is None or not cls._ack_matches(result[0], candidate):
            return None
        try:
            return AckStatus(int(result[0]["status"][0]))
        except ValueError:
            return None

    def wait_applied(self, candidate: ActionCandidate, *, timeout_s: float = 0.5) -> bool:
        """Wait for every enabled actuator to acknowledge SDK application."""
        deadline = time.monotonic() + timeout_s
        failed = {AckStatus.REJECTED, AckStatus.SDK_FAILED}
        while time.monotonic() < deadline:
            arm_status = (
                AckStatus.APPLIED
                if candidate.arm_qpos is None
                else self._ack_status(self.shared.arm_ack_ring, candidate)
            )
            hand_status = (
                AckStatus.APPLIED
                if candidate.hand_qpos is None
                else self._ack_status(self.shared.hand_ack_ring, candidate)
            )
            if arm_status in failed or hand_status in failed:
                return False
            if arm_status is AckStatus.APPLIED and hand_status is AckStatus.APPLIED:
                return True
            time.sleep(0.001)
        logger.error("SafeCommandPublisher: apply timeout for action_id=%d", candidate.action_id)
        return False


def publish_joint_targets(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    is_hold: bool = False,
    prepare_timeout_s: float = 0.06,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    dt_s: float | None = None,
    safety_gate: ActionSafetyGate | None = None,
    wait_applied: bool = False,
    apply_timeout_s: float = 0.5,
) -> ActionCandidate | None:
    """Gate, prepare, and commit one correlated endpoint.

    Every producer must supply the canonical geometry-aware gate.  Missing
    workspace, table, or transition-collision callbacks are rejected here;
    this function never constructs a reduced-check fallback gate.

    Returns the exact candidate committed to the workers (including any
    dt-aware clamp), or ``None`` when validation/publication fails.
    """
    if safety_gate is None or not safety_gate.config.require_geometry_checks:
        logger.error("joint target rejected: an explicit geometry-aware ActionSafetyGate is required")
        return None
    gate = safety_gate

    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = time.monotonic_ns()
    lead_time_s = float(shared.action_lead_time_s)
    validity_s = float(shared.action_validity_s)
    control_hz = float(shared.action_control_hz)
    target_ns = now_ns + int(lead_time_s * 1e9)
    candidate = ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        session_generation=int(shared.session_generation.value),
        policy_epoch=int(shared.policy_epoch.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=target_ns,
        valid_until_monotonic_ns=target_ns + int(validity_s * 1e9),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
        hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
        chunk_id=action_id,
        step_index=0,
        is_hold=is_hold,
    )
    try:
        arm_result = shared.arm_state_ring.read_latest()
        if arm_result is None:
            raise ValueError("arm feedback unavailable for safety gate")
        arm_record = arm_result[0][0]
        if "state_valid" in arm_record.dtype.names and not bool(arm_record["state_valid"]):
            raise ValueError("arm feedback is invalid")
        if "source_monotonic_ns" in arm_record.dtype.names:
            arm_source_ns = int(arm_record["source_monotonic_ns"])
            arm_age_ns = now_ns - arm_source_ns
            if arm_source_ns <= 0 or arm_age_ns < 0 or arm_age_ns > int(gate.config.observation_max_age_s * 1e9):
                raise ValueError("arm feedback is stale or from the future")
        current_arm = np.asarray(arm_record["qpos"], dtype=np.float64)
        current_hand = np.asarray(shared.hand_home_qpos_rad, dtype=np.float64)
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is not None:
            hand_record = hand_result[0][0]
            if "state_valid" in hand_record.dtype.names and not bool(hand_record["state_valid"]):
                raise ValueError("hand feedback is invalid")
            else:
                if "source_monotonic_ns" in hand_record.dtype.names:
                    hand_source_ns = int(hand_record["source_monotonic_ns"])
                    hand_age_ns = now_ns - hand_source_ns
                    if (
                        hand_source_ns <= 0
                        or hand_age_ns < 0
                        or hand_age_ns > int(gate.config.observation_max_age_s * 1e9)
                    ):
                        raise ValueError("hand feedback is stale or from the future")
                    else:
                        current_hand = np.asarray(hand_record["qpos"], dtype=np.float64)
                else:
                    current_hand = np.asarray(hand_record["qpos"], dtype=np.float64)
        elif hand_qpos is not None:
            raise ValueError("hand feedback unavailable for safety gate")

        feedback_source_times = [arm_source_ns] if "source_monotonic_ns" in arm_record.dtype.names else []
        if hand_qpos is not None and hand_result is not None and "source_monotonic_ns" in hand_record.dtype.names:
            feedback_source_times.append(hand_source_ns)
        snapshot_anchor_ns = (
            min(feedback_source_times)
            if observation_anchor_monotonic_ns is None and feedback_source_times
            else now_ns if observation_anchor_monotonic_ns is None else int(observation_anchor_monotonic_ns)
        )
        empty = FrozenArrayMap(())
        snapshot = ObservationSnapshot(
            observation_id=candidate.observation_id,
            anchor_monotonic_ns=snapshot_anchor_ns,
            values=empty,
            source_monotonic_ns=empty,
            publish_monotonic_ns=empty,
            valid_history_mask=empty,
            session_generation=candidate.session_generation,
        )
        gate_result = gate.evaluate(
            candidate,
            snapshot=snapshot,
            current_arm_qpos=current_arm,
            current_hand_qpos=current_hand,
            expected_session_generation=candidate.session_generation,
            expected_policy_epoch=candidate.policy_epoch,
            now_monotonic_ns=now_ns,
            dt_s=(1.0 / control_hz if dt_s is None else float(dt_s)),
        )
        if not gate_result.accepted or gate_result.candidate is None:
            logger.warning("joint target rejected by ActionSafetyGate: %s", gate_result.reason)
            return None
        publisher = SafeCommandPublisher(shared)
        if not publisher.publish(gate_result.candidate, prepare_timeout_s=prepare_timeout_s):
            return None
        if wait_applied and not publisher.wait_applied(gate_result.candidate, timeout_s=apply_timeout_s):
            return None
        return gate_result.candidate
    except (TimeoutError, ValueError):
        logger.warning("joint target publication failed", exc_info=True)
        return None


def quiesce_for_policy_restart(
    shared: Any,
    *,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    safety_gate: ActionSafetyGate,
    timeout_s: float = 0.5,
) -> int:
    """Invalidate the old epoch and reach an applied coordinated hold.

    The caller may load/warm a replacement backend only after this returns.
    This function deliberately leaves the system ARMED; returning to RUNNING
    remains an explicit operator action, so there is no seamless hot swap.
    """
    from dexmani_real.robot.safety import SafetyState, transition

    transition(shared, SafetyState.ARMED)
    with shared.policy_epoch.get_lock():
        shared.policy_epoch.value = int(shared.policy_epoch.value) + 1
        epoch = int(shared.policy_epoch.value)
    if not publish_joint_targets(
        shared,
        arm_qpos,
        hand_qpos,
        is_hold=True,
        safety_gate=safety_gate,
        wait_applied=True,
        apply_timeout_s=timeout_s,
    ):
        shared.error_state.value = True
        raise RuntimeError("policy restart failed to reach an applied coordinated hold")
    return epoch


class JointActionScheduler:
    """Policy-owned chunk overlap, replacement, expiry, and per-tick scheduler."""

    def __init__(self, action_spec: ActionSpec) -> None:
        self.action_spec = action_spec
        self._future: list[ActionCandidate] = []
        self._committed: set[int] = set()
        self._all_late = False

    def submit(self, chunk: ActionChunk, *, now_monotonic_ns: int | None = None) -> None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        # Preserve committed endpoints; replace every uncommitted future step.
        self._future = [step for step in self._future if step.action_id in self._committed]
        accepted = [step for step in chunk.steps if step.valid_until_monotonic_ns >= now_ns]
        self._all_late = not accepted
        self._future.extend(accepted)
        self._future.sort(key=lambda step: (step.target_monotonic_ns, step.action_id))

    def mark_committed(self, action_id: int) -> None:
        self._committed.add(int(action_id))

    def reset(self) -> None:
        """Invalidate all scheduled endpoints at an epoch boundary."""
        self._future.clear()
        self._committed.clear()
        self._all_late = False

    def pop_due(self, *, now_monotonic_ns: int | None = None) -> ActionCandidate | None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        self._future = [step for step in self._future if step.valid_until_monotonic_ns >= now_ns]
        due = [step for step in self._future if step.target_monotonic_ns <= now_ns]
        if not due:
            return None
        selected = due[-1]
        self._future = [step for step in self._future if step.action_id != selected.action_id]
        self._committed.discard(selected.action_id)
        return selected

    def pop_ready(
        self,
        *,
        lead_time_s: float,
        now_monotonic_ns: int | None = None,
    ) -> ActionCandidate | None:
        """Pop the newest endpoint whose prepare window has opened.

        ``target_monotonic_ns`` is the worker application time, so a
        coordinator must publish before it becomes due.  This method keeps the
        older :meth:`pop_due` behavior available for non-actuator consumers.
        """
        if not np.isfinite(lead_time_s) or lead_time_s < 0:
            raise ValueError("lead_time_s must be finite and non-negative")
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        self._future = [step for step in self._future if step.valid_until_monotonic_ns >= now_ns]
        ready = [step for step in self._future if step.target_monotonic_ns <= now_ns + int(lead_time_s * 1e9)]
        if not ready:
            return None
        selected = ready[-1]
        self._future = [step for step in self._future if step.action_id != selected.action_id]
        self._committed.discard(selected.action_id)
        return selected

    @property
    def pending(self) -> tuple[ActionCandidate, ...]:
        return tuple(self._future)

    @property
    def all_late(self) -> bool:
        """Whether the most recent chunk had no endpoint that remained valid."""
        return self._all_late

    def make_coordinated_hold(
        self,
        *,
        template: ActionCandidate,
        arm_qpos: np.ndarray,
        hand_qpos: np.ndarray | None,
        action_id: int,
        now_monotonic_ns: int | None = None,
    ) -> ActionCandidate:
        """Create a fresh correlated hold after an all-late chunk."""
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        lead_ns = int(2 * self.action_spec.dt_s * 1e9)
        target_ns = now_ns + lead_ns
        return ActionCandidate(
            observation_id=template.observation_id,
            session_generation=template.session_generation,
            policy_epoch=template.policy_epoch,
            action_id=int(action_id),
            created_monotonic_ns=now_ns,
            target_monotonic_ns=target_ns,
            valid_until_monotonic_ns=target_ns + int(self.action_spec.dt_s * 1e9),
            arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
            hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
            chunk_id=template.chunk_id,
            step_index=template.step_index,
            is_hold=True,
        )

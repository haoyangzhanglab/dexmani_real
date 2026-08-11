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

from dexmani_real.config.defaults import hand, policy
from dexmani_real.utils.schema import (
    ACK_DTYPE,
    ARM_COMMAND_DTYPE,
    ARM_JOINT_SHAPE,
    ARM_STATE_DTYPE,
    COMMIT_DTYPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
)
from dexmani_real.policy.runtime import ActionCandidate, ActionChunk, ActionSpec, FrozenArrayMap, ObservationSnapshot
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_HAND_HOME_PUBLISH_INTERVAL_S = 0.1


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


def advance_policy_epoch(shared: Any) -> int:
    """Invalidate commands prepared under the previous policy epoch."""
    lock_getter = getattr(shared.policy_epoch, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            shared.policy_epoch.value = int(shared.policy_epoch.value) + 1
            return int(shared.policy_epoch.value)
    shared.policy_epoch.value = int(shared.policy_epoch.value) + 1
    return int(shared.policy_epoch.value)


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
        if arm_lower.shape != ARM_JOINT_SHAPE or arm_upper.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm gate limits must have seven entries")
        if hand_lower.shape != HAND_JOINT_SHAPE or hand_upper.shape != HAND_JOINT_SHAPE:
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
            candidate.target_monotonic_ns <= now_ns
            or candidate.valid_until_monotonic_ns < now_ns
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
        if arm_start.shape != ARM_JOINT_SHAPE or hand_start.shape != HAND_JOINT_SHAPE:
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
            arm_end.shape != ARM_JOINT_SHAPE
            or hand_end.shape != HAND_JOINT_SHAPE
            or not np.all(np.isfinite(arm_end))
            or not np.all(np.isfinite(hand_end))
        ):
            return GateResult(False, None, "invalid candidate joint shape/values")

        arm_low = np.asarray(self.config.arm_joint_lower_rad, dtype=np.float64)
        arm_high = np.asarray(self.config.arm_joint_upper_rad, dtype=np.float64)
        hand_low = np.asarray(self.config.hand_joint_lower_rad, dtype=np.float64)
        hand_high = np.asarray(self.config.hand_joint_upper_rad, dtype=np.float64)
        # Only check joint limits on explicitly commanded targets.
        # When an actuator is not commanded (qpos is None -> hold in place),
        # the gate enforces command correctness, not sensor precision -
        # measured feedback may sit outside nominal limits due to encoder
        # resolution, PID steady-state error, or external load.
        if candidate.arm_qpos is not None and (
            np.any(arm_end < arm_low) or np.any(arm_end > arm_high)
        ):
            return GateResult(False, None, "arm joint limit violation")
        if candidate.hand_qpos is not None and (
            np.any(hand_end < hand_low) or np.any(hand_end > hand_high)
        ):
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
    enable_table_check: bool = True,
) -> ActionSafetyGate:
    """Build the canonical geometry-aware gate around a configured planner.

    The table check samples the conservative Cartesian product of arm and hand
    progress because the two workers can apply a committed endpoint up to one
    worker tick apart.  Planner state is restored to the measured hand pose on
    rejection and advanced to the accepted endpoint on success.

    Set *enable_table_check* to False for teleoperation where the operator
    provides the table-awareness; ``send_arm_home`` performs its own
    independent table-clearance validation.
    """

    _table_check: Callable[..., bool] | None = None
    if enable_table_check:
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

        _table_check = table_clearance_check

    return ActionSafetyGate(
        config,
        workspace_check=planner.is_workspace_segment_safe,
        transition_collision_check=planner.collision_model.check_transition_collision_free,
        table_clearance_check=_table_check,
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
    return int(command["created_monotonic_ns"][0]) <= committed_ns < int(command["target_monotonic_ns"][0])


def _source_age_s(source_monotonic_ns: int, *, now_monotonic_ns: int, max_age_s: float, actuator: str) -> float:
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("feedback max age must be finite and positive")
    if source_monotonic_ns <= 0:
        raise ValueError(f"{actuator} feedback has no source timestamp")
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0 or age_s > max_age_s:
        raise ValueError(f"{actuator} feedback is stale or from the future")
    return age_s


def _validated_arm_feedback_qpos(record: np.void, *, now_monotonic_ns: int, max_age_s: float) -> np.ndarray:
    """Return a private copy of complete, healthy arm feedback."""
    if not isinstance(record, np.void) or record.dtype != ARM_STATE_DTYPE:
        raise ValueError("arm feedback has an invalid schema")
    if not bool(record["connected"]) or not bool(record["state_valid"]):
        raise ValueError("arm feedback is disconnected or invalid")
    if int(record["error_code"]) != 0:
        raise ValueError(f"arm feedback reports controller error {int(record['error_code'])}")
    _source_age_s(
        int(record["source_monotonic_ns"]),
        now_monotonic_ns=now_monotonic_ns,
        max_age_s=max_age_s,
        actuator="arm",
    )
    for name, expected_shape in (
        ("qpos", ARM_JOINT_SHAPE),
        ("qvel", ARM_JOINT_SHAPE),
        ("eef_pos", (3,)),
        ("eef_rot6d", (6,)),
    ):
        value = np.asarray(record[name], dtype=np.float64)
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"arm feedback {name} has an invalid shape or value")
    return np.asarray(record["qpos"], dtype=np.float64).copy()


def _validated_hand_feedback_qpos(record: np.void, *, now_monotonic_ns: int, max_age_s: float) -> np.ndarray:
    """Return a private copy of complete, healthy measured-hand feedback."""
    if not isinstance(record, np.void) or record.dtype != HAND_STATE_DTYPE:
        raise ValueError("hand feedback has an invalid schema")
    if not bool(record["connected"]) or not bool(record["state_valid"]):
        raise ValueError("hand feedback is disconnected or invalid")
    if bool(record["error_state"]):
        raise ValueError("hand feedback reports a hardware error")
    if bool(record["qpos_stale"]):
        raise ValueError("hand joint feedback is stale")
    if not bool(record["send_healthy"]) or not bool(record["read_healthy"]):
        raise ValueError("hand command/state I/O is unhealthy")
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        value = np.asarray(record[name])
        if value.shape != HAND_JOINT_SHAPE or np.any(value != 0):
            raise ValueError(f"hand feedback reports {name}")
    _source_age_s(
        int(record["source_monotonic_ns"]),
        now_monotonic_ns=now_monotonic_ns,
        max_age_s=max_age_s,
        actuator="hand",
    )
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
        raise ValueError("hand feedback qpos has an invalid shape or value")
    return qpos.copy()


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

    def prepare(self, candidate: ActionCandidate, *, timeout_s: float = policy.action_prepare_timeout_s) -> None:
        if candidate.arm_qpos is not None:
            arm_frame = make_command_frame(candidate, actuator="arm")
            try:
                self.shared.arm_action_q.put(arm_frame, block=True, timeout=timeout_s)
            except Full as exc:
                raise TimeoutError("arm action queue full") from exc
        if candidate.hand_qpos is not None:
            self.shared.hand_cmd_ring.write(make_command_frame(candidate, actuator="hand"))

    def commit(self, candidate: ActionCandidate) -> bool:
        committed_ns = time.monotonic_ns()
        if (
            committed_ns < candidate.created_monotonic_ns
            or committed_ns >= candidate.target_monotonic_ns
            or committed_ns > candidate.valid_until_monotonic_ns
        ):
            logger.error("SafeCommandPublisher: refusing late commit for action_id=%d", candidate.action_id)
            return False
        frame = np.zeros(1, dtype=COMMIT_DTYPE)
        frame["session_generation"][0] = candidate.session_generation
        frame["policy_epoch"][0] = candidate.policy_epoch
        frame["observation_id"][0] = candidate.observation_id
        frame["action_id"][0] = candidate.action_id
        frame["chunk_id"][0] = candidate.chunk_id
        frame["step_index"][0] = candidate.step_index
        frame["created_monotonic_ns"][0] = candidate.created_monotonic_ns
        frame["committed_monotonic_ns"][0] = committed_ns
        frame["target_monotonic_ns"][0] = candidate.target_monotonic_ns
        frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
        frame["is_hold"][0] = candidate.is_hold
        self.shared.action_commit_ring.write(frame)
        return True

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

    def publish(
        self,
        candidate: ActionCandidate,
        *,
        prepare_timeout_s: float = policy.action_prepare_timeout_s,
    ) -> bool:
        """Prepare enabled actuators and commit before one shared deadline."""
        if not np.isfinite(prepare_timeout_s) or prepare_timeout_s <= 0.0:
            raise ValueError("prepare_timeout_s must be finite and positive")
        started_ns = time.monotonic_ns()
        deadline_ns = min(
            started_ns + int(prepare_timeout_s * 1e9),
            candidate.target_monotonic_ns,
            candidate.valid_until_monotonic_ns,
        )
        remaining_s = (deadline_ns - time.monotonic_ns()) * 1e-9
        if remaining_s <= 0.0:
            logger.error("SafeCommandPublisher: action_id=%d has no prepare window", candidate.action_id)
            return False
        try:
            self.prepare(candidate, timeout_s=remaining_s)
        except TimeoutError:
            logger.error("SafeCommandPublisher: prepare enqueue timeout for action_id=%d", candidate.action_id)
            return False
        while time.monotonic_ns() < deadline_ns:
            arm_ready = candidate.arm_qpos is None or self._prepared(self.shared.arm_ack_ring, candidate)
            hand_ready = candidate.hand_qpos is None or self._prepared(self.shared.hand_ack_ring, candidate)
            if arm_ready and hand_ready:
                return self.commit(candidate)
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

    def wait_applied(self, candidate: ActionCandidate, *, timeout_s: float = policy.action_apply_timeout_s) -> bool:
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
    prepare_timeout_s: float = policy.action_prepare_timeout_s,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    dt_s: float | None = None,
    safety_gate: ActionSafetyGate | None = None,
    wait_applied: bool = False,
    apply_timeout_s: float = policy.action_apply_timeout_s,
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
        current_arm = _validated_arm_feedback_qpos(
            arm_record,
            now_monotonic_ns=now_ns,
            max_age_s=gate.config.observation_max_age_s,
        )
        arm_source_ns = int(arm_record["source_monotonic_ns"])
        current_hand = np.asarray(shared.hand_home_qpos_rad, dtype=np.float64)
        hand_result = shared.hand_state_ring.read_latest()
        hand_source_ns: int | None = None
        if hand_result is not None:
            hand_record = hand_result[0][0]
            current_hand = _validated_hand_feedback_qpos(
                hand_record,
                now_monotonic_ns=now_ns,
                max_age_s=gate.config.observation_max_age_s,
            )
            hand_source_ns = int(hand_record["source_monotonic_ns"])
        elif hand_qpos is not None:
            raise ValueError("hand feedback unavailable for safety gate")

        feedback_source_times = [arm_source_ns]
        if hand_source_ns is not None:
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
        # Re-anchor timestamps after gate evaluation so that collision checks
        # and feedback validation do not consume the prepare window.  The
        # candidate identity (action_id, session_generation, policy_epoch) is
        # unchanged - only the delivery schedule is refreshed.
        _publish_ns = time.monotonic_ns()
        _lead_ns = int(lead_time_s * 1e9)
        _validity_ns = int(validity_s * 1e9)
        _refreshed = replace(
            gate_result.candidate,
            target_monotonic_ns=_publish_ns + _lead_ns,
            valid_until_monotonic_ns=_publish_ns + _lead_ns + _validity_ns,
        )
        publisher = SafeCommandPublisher(shared)
        if not publisher.publish(_refreshed, prepare_timeout_s=prepare_timeout_s):
            return None
        if wait_applied and not publisher.wait_applied(_refreshed, timeout_s=apply_timeout_s):
            return None
        return _refreshed
    except (TimeoutError, ValueError):
        logger.warning("joint target publication failed", exc_info=True)
        return None


def write_hand_cmd(shared: Any, qpos: np.ndarray, *, safety_gate: ActionSafetyGate | None = None) -> bool:
    """Publish a hand target together with a measured arm hold."""
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        return False
    arm_qpos = np.asarray(arm_result[0]["qpos"][0], dtype=np.float64)
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


def hand_home_converge(
    shared: Any,
    home_qpos: np.ndarray,
    *,
    timeout_s: float = hand.home_settle_timeout_s,
    tol_rad: float = hand.home_settle_tol_rad,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    safety_gate: ActionSafetyGate | None = None,
    abort_requested: Callable[[], bool] | None = None,
) -> tuple[bool, np.ndarray | None]:
    """Publish coordinated endpoints until fresh hand feedback reaches home."""
    if not np.isfinite(timeout_s) or timeout_s <= 0 or not np.isfinite(tol_rad) or tol_rad <= 0:
        raise ValueError("hand home timeout and tolerance must be finite and positive")
    home_qpos = np.asarray(home_qpos, dtype=np.float64)
    if home_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(home_qpos)):
        raise ValueError(f"hand home must be a finite array with shape {HAND_JOINT_SHAPE}")
    deadline = time.monotonic() + timeout_s
    requested_after_ns: int | None = None
    first = True

    def aborted() -> bool:
        return bool(
            getattr(getattr(shared, "estop_request", None), "value", False)
            or getattr(getattr(shared, "error_state", None), "value", False)
            or (abort_requested is not None and abort_requested())
        )

    while time.monotonic() < deadline:
        if aborted():
            return False, None
        if check_is_running and not shared.is_running.value:
            break
        if heartbeat:
            shared.policy_heartbeat_s.value = time.monotonic()
        if not write_hand_cmd(shared, home_qpos, safety_gate=safety_gate):
            if verbose:
                print("  hand: coordinated home command was rejected", flush=True)
            return False, None
        if requested_after_ns is None:
            requested_after_ns = time.monotonic_ns()
        if aborted():
            return False, None
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is not None:
            state = hand_result[0][0]
            now_ns = time.monotonic_ns()
            try:
                current = _validated_hand_feedback_qpos(
                    state,
                    now_monotonic_ns=now_ns,
                    max_age_s=timeout_s,
                )
            except ValueError:
                current = None
            source_ns = int(state["source_monotonic_ns"])
            if current is not None and requested_after_ns <= source_ns <= now_ns:
                err = float(np.max(np.abs(current - home_qpos)))
                if err < tol_rad:
                    if aborted():
                        return False, None
                    if verbose:
                        print("  hand: home reached", flush=True)
                    return True, current.copy()
                if verbose and first:
                    print(f"  hand: homing... (max_err={np.rad2deg(err):.0f}°)", flush=True)
                    first = False
        # Allow the two-worker lead time plus one actuator tick before
        # replacing the next latest-wins hand endpoint.
        time.sleep(_HAND_HOME_PUBLISH_INTERVAL_S)

    if verbose:
        print(f"  hand: home not confirmed after {timeout_s:.0f}s", flush=True)
    return False, None


class JointActionScheduler:
    """Policy-owned chunk overlap, replacement, expiry, and per-tick scheduler."""

    def __init__(self, action_spec: ActionSpec) -> None:
        self.action_spec = action_spec
        self._future: list[ActionCandidate] = []
        self._all_late = False

    def submit(self, chunk: ActionChunk, *, now_monotonic_ns: int | None = None) -> None:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        # Already-published endpoints have been popped. A newer chunk replaces
        # every endpoint that has not yet been published.
        self._future.clear()
        accepted = [
            step
            for step in chunk.steps
            if step.target_monotonic_ns > now_ns and step.valid_until_monotonic_ns >= now_ns
        ]
        self._all_late = not accepted
        self._future.extend(accepted)
        self._future.sort(key=lambda step: (step.target_monotonic_ns, step.action_id))

    def reset(self) -> None:
        """Invalidate all scheduled endpoints at an epoch boundary."""
        self._future.clear()
        self._all_late = False

    def pop_ready(
        self,
        *,
        lead_time_s: float,
        now_monotonic_ns: int | None = None,
    ) -> ActionCandidate | None:
        """Pop the earliest endpoint whose prepare window has opened.

        ``target_monotonic_ns`` is the worker application time, so a
        coordinator must publish before it becomes due.
        """
        if not np.isfinite(lead_time_s) or lead_time_s < 0:
            raise ValueError("lead_time_s must be finite and non-negative")
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        late = [
            step
            for step in self._future
            if step.target_monotonic_ns <= now_ns or step.valid_until_monotonic_ns < now_ns
        ]
        self._future = [
            step
            for step in self._future
            if step.target_monotonic_ns > now_ns and step.valid_until_monotonic_ns >= now_ns
        ]
        if late and not self._future:
            self._all_late = True
        ready = [step for step in self._future if step.target_monotonic_ns <= now_ns + int(lead_time_s * 1e9)]
        if not ready:
            return None
        # Workers enforce monotonically increasing action IDs.  Selecting the
        # newest ready endpoint while leaving older ready endpoints queued would
        # publish those older IDs later and make the workers reject them as
        # OUT_OF_ORDER after any coordinator stall spanning multiple steps.
        selected = ready[0]
        self._future = [step for step in self._future if step.action_id != selected.action_id]
        return selected

    @property
    def pending(self) -> tuple[ActionCandidate, ...]:
        return tuple(self._future)

    @property
    def all_late(self) -> bool:
        """Whether the most recent chunk has no endpoint publishable before target."""
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

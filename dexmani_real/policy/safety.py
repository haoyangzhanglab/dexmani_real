"""Unified safety gate — the single validation boundary for all action paths.

Every coordinator (teleop, keyboard, replay, calibration) must
route candidates through :class:`SafetyGate` before :func:`send_command` writes
to the actuator IPC primitives.  Workers trust the gate and apply commands
immediately with hardware-boundary checks (safety state, dtype, finite values,
command/mechanical limits, lifecycle metadata, and SDK return code).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_JOINT_SHAPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
)

logger = get_logger(__name__)


def advance_run_generation(shared: Any) -> int:
    """Invalidate candidates prepared before the current control run."""
    lock_getter = getattr(shared.run_generation, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            shared.run_generation.value = int(shared.run_generation.value) + 1
            return int(shared.run_generation.value)
    shared.run_generation.value = int(shared.run_generation.value) + 1
    return int(shared.run_generation.value)


@dataclass(frozen=True)
class GateResult:
    """Outcome of :meth:`SafetyGate.validate`."""

    accepted: bool
    candidate: Any  # ActionCandidate (lazy import to avoid cycle)
    reason: str = ""


class SafetyGate:
    """Single validation boundary for all action paths.

    Pipeline (short-circuit, fail-closed):

    1. **Well-formed** — representation, shapes, finite values
    2. **Joint limits** — commanded actuators only; hold actuators skip
    3. **Workspace** — optional segment check

    Velocity envelope checking was removed (2026-08-12).  xArm Mode 6
    firmware enforces velocity, acceleration, and collision limits as the
    final backstop.  Collision and transition geometry checks were
    removed (2026-08-12); collision-free homing paths are planned
    independently through ``plan_joint_home_path`` /
    ``plan_band_alignment_path``, which call the collision model directly
    rather than through the SafetyGate.

    The controller validates ``run_generation`` and the proposal validity
    window before publication.  Workers re-check generation and expiry before
    applying the resulting fixed command.  The gate never silently converts a
    rejected motion into a different joint-space endpoint.
    """

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
    ) -> None:
        _arm_low = np.asarray(arm_joint_lower_rad, dtype=np.float64)
        _arm_high = np.asarray(arm_joint_upper_rad, dtype=np.float64)
        _hand_low = np.asarray(hand_joint_lower_rad, dtype=np.float64)
        _hand_high = np.asarray(hand_joint_upper_rad, dtype=np.float64)
        if _arm_low.shape != ARM_JOINT_SHAPE or _arm_high.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm joint limits must have seven entries")
        if _hand_low.shape != HAND_JOINT_SHAPE or _hand_high.shape != HAND_JOINT_SHAPE:
            raise ValueError("hand joint limits must have twelve entries")
        concat = np.concatenate((_arm_low, _arm_high, _hand_low, _hand_high))
        if (
            not np.all(np.isfinite(concat))
            or np.any(_arm_low > _arm_high)
            or np.any(_hand_low > _hand_high)
        ):
            raise ValueError("joint limits must be finite and ordered")

        self.arm_low = _arm_low
        self.arm_high = _arm_high
        self.hand_low = _hand_low
        self.hand_high = _hand_high

    # -- callback set after construction (avoids circular imports) ----------
    workspace_check: Any = None  # Callable[[np.ndarray, np.ndarray], bool] | None

    def validate(
        self,
        candidate: Any,  # ActionCandidate
        *,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
        dt_s: float,
        run_generation: int,
    ) -> GateResult:
        """Run the full validation pipeline.

        Args:
            candidate: The proposed ``ActionCandidate``.
            current_arm_qpos: Latest measured arm joint positions [rad].
            current_hand_qpos: Latest measured hand joint positions [rad].
            dt_s: Action period (``1 / control_hz``). Retained for API
                stability; unused after velocity-envelope removal (2026-08-12).
            run_generation: Expected control-run generation.

        Returns:
            ``GateResult`` with the unchanged accepted candidate.
        """
        # 1 ── Well-formed ────────────────────────────────────────────
        if (
            candidate.representation != "joint_position"
            or candidate.units != "rad"
            or candidate.frame != "robot_joint"
        ):
            return GateResult(
                False, candidate, "unsupported representation/units/frame"
            )

        if candidate.run_generation != run_generation:
            return GateResult(False, candidate, "run generation mismatch")

        arm_start = np.asarray(current_arm_qpos, dtype=np.float64)
        hand_start = np.asarray(current_hand_qpos, dtype=np.float64)
        if arm_start.shape != ARM_JOINT_SHAPE or hand_start.shape != HAND_JOINT_SHAPE:
            return GateResult(False, candidate, "invalid current joint state shape")
        if not np.all(np.isfinite(arm_start)) or not np.all(np.isfinite(hand_start)):
            return GateResult(False, candidate, "current joint state contains NaN/Inf")

        arm_end = (
            arm_start.copy()
            if candidate.arm_qpos is None
            else np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        )
        hand_end = (
            hand_start.copy()
            if candidate.hand_qpos is None
            else np.asarray(candidate.hand_qpos, dtype=np.float64).copy()
        )
        if arm_end.shape != ARM_JOINT_SHAPE or hand_end.shape != HAND_JOINT_SHAPE:
            return GateResult(False, candidate, "invalid candidate joint shape")
        if not np.all(np.isfinite(arm_end)) or not np.all(np.isfinite(hand_end)):
            return GateResult(False, candidate, "candidate contains NaN/Inf")

        # 2 ── Joint limits (commanded actuators only) ─────────────────
        if candidate.arm_qpos is not None and (
            np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)
        ):
            return GateResult(False, candidate, "arm joint limit violation")
        if candidate.hand_qpos is not None and (
            np.any(hand_end < self.hand_low - 1e-12)
            or np.any(hand_end > self.hand_high + 1e-12)
        ):
            return GateResult(False, candidate, "hand joint limit violation")

        # (removed) Velocity envelope ────────────────────────────────────
        # Per the xArm Mode 6 contract the firmware is the final velocity,
        # acceleration, and collision backstop.  The software gate no longer
        # rejects on command-to-command or command-to-measured speed so that
        # Cartesian keyboard deltas never deadlock when the IK maps a fixed
        # step to joint changes above the configured limit.

        # 3 ── Workspace (optional) ────────────────────────────────────
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, arm_end):
                    return GateResult(False, candidate, "workspace")
            except Exception:
                logger.warning(
                    "SafetyGate: workspace check failed closed", exc_info=True
                )
                return GateResult(False, candidate, "workspace check failed")

        return GateResult(True, candidate)


# ---------------------------------------------------------------------------
# Command serialization and publication
# ---------------------------------------------------------------------------


def _make_arm_command(
    candidate: Any, now_monotonic_ns: int, target_monotonic_ns: int
) -> np.ndarray:
    """Serialize an ActionCandidate into an ARM_COMMAND_DTYPE record."""
    if candidate.arm_qpos is None:
        raise ValueError("candidate has no arm command")
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = target_monotonic_ns + int(3e8)  # +300ms
    frame["is_hold"][0] = int(bool(candidate.is_hold))
    frame["qpos_cmd"][0] = candidate.arm_qpos
    return frame


def _make_hand_command(
    candidate: Any, now_monotonic_ns: int, target_monotonic_ns: int
) -> np.ndarray:
    """Serialize an ActionCandidate into a HAND_COMMAND_DTYPE record."""
    if candidate.hand_qpos is None:
        raise ValueError("candidate has no hand command")
    frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = target_monotonic_ns + int(3e8)
    frame["is_hold"][0] = int(bool(candidate.is_hold))
    frame["qpos_cmd"][0] = candidate.hand_qpos
    return frame


def send_command(
    shared: Any,
    candidate: Any,  # ActionCandidate
    *,
    prepare_timeout_s: float | None = None,
) -> bool:
    """Publish arm and/or hand commands to the actuator IPC primitives.

    Fire-and-forget: the arm command goes into the bounded queue, the hand
    command overwrites the latest-wins ring.  No ACKs, no commit protocol.

    Returns:
        ``True`` when every enabled actuator's command was accepted by the
        transport.  ``False`` when the arm queue is full (coordinator should
        reject or enter command quiescence) or when the candidate's temporal
        window has already closed.
    """
    timeout = (
        prepare_timeout_s
        if prepare_timeout_s is not None
        else policy_defaults.action_prepare_timeout_s
    )
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("prepare_timeout_s must be finite and positive")

    now_ns = time.monotonic_ns()
    lead_time_s = float(getattr(shared, "action_lead_time_s", 0.05))
    target_ns = now_ns + int(lead_time_s * 1e9)

    # The candidate validity window protects the policy boundary.  The publish
    # boundary owns the worker delivery target because queueing adds latency.
    if target_ns <= now_ns or candidate.valid_until_monotonic_ns < now_ns:
        logger.error(
            "send_command: action_id=%d temporal window closed", candidate.action_id
        )
        return False

    deadline_ns = now_ns + int(timeout * 1e9)
    remaining_s = (deadline_ns - time.monotonic_ns()) * 1e-9
    if remaining_s <= 0:
        return False

    if candidate.arm_qpos is not None:
        try:
            arm_frame = _make_arm_command(candidate, now_ns, target_ns)
            shared.arm_action_q.put(arm_frame, block=True, timeout=remaining_s)
        except Full:
            logger.warning(
                "send_command: arm endpoint queue backpressure (action_id=%d)",
                candidate.action_id,
            )
            return False

    if candidate.hand_qpos is not None:
        hand_frame = _make_hand_command(candidate, now_ns, target_ns)
        shared.hand_cmd_ring.write(hand_frame)

    return True


# ---------------------------------------------------------------------------
# Worker-side validation (minimal — trust the gate)
# ---------------------------------------------------------------------------


def _worker_command_is_current(
    command: np.ndarray,
    *,
    expected_run_generation: int | None,
    now_monotonic_ns: int | None,
) -> bool:
    """Validate the lifecycle metadata common to fixed actuator commands."""
    if expected_run_generation is not None and int(command["run_generation"][0]) != int(
        expected_run_generation
    ):
        return False
    if now_monotonic_ns is not None:
        valid_until_ns = int(command["valid_until_monotonic_ns"][0])
        if valid_until_ns <= 0 or int(now_monotonic_ns) > valid_until_ns:
            return False
    return True


def worker_validate_arm(
    command: np.ndarray,
    *,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Minimal hardware-level check for an arm command from the queue.

    Returns True when the command is well-formed, belongs to the active run,
    and has not expired. The gate already validated limits and geometry.
    """
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == ARM_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    )
    return bool(
        well_formed
        and _worker_command_is_current(
            command,
            expected_run_generation=expected_run_generation,
            now_monotonic_ns=now_monotonic_ns,
        )
    )


def worker_validate_hand(
    command: np.ndarray,
    *,
    qpos_lower_rad: np.ndarray,
    qpos_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    previous_qpos_cmd: np.ndarray | None = None,
    max_command_delta_rad: float | np.ndarray | None = None,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Hardware-boundary check for a hand command from the ring.

    Returns True when the command is well-formed, belongs to the active run,
    has not expired, and lies inside both operational and rated mechanical
    limits. When supplied, the last SDK-accepted target and delta limit are
    checked command-to-command as well. These redundant checks protect
    direct/home publishers and IPC corruption; they never modify the endpoint.
    """
    command_lower = np.asarray(qpos_lower_rad, dtype=np.float64)
    command_upper = np.asarray(qpos_upper_rad, dtype=np.float64)
    mechanical_lower = np.asarray(mechanical_lower_rad, dtype=np.float64)
    mechanical_upper = np.asarray(mechanical_upper_rad, dtype=np.float64)
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
    limits_well_formed = (
        command_lower.shape == HAND_JOINT_SHAPE
        and command_upper.shape == HAND_JOINT_SHAPE
        and mechanical_lower.shape == HAND_JOINT_SHAPE
        and mechanical_upper.shape == HAND_JOINT_SHAPE
        and np.all(
            np.isfinite(
                np.concatenate(
                    (command_lower, command_upper, mechanical_lower, mechanical_upper)
                )
            )
        )
        and np.all(command_lower <= command_upper)
        and np.all(mechanical_lower <= mechanical_upper)
        and np.all(mechanical_lower >= rated_lower)
        and np.all(mechanical_upper <= rated_upper)
        and np.all(command_lower >= mechanical_lower)
        and np.all(command_upper <= mechanical_upper)
    )
    qpos_cmd = (
        np.asarray(command["qpos_cmd"][0], dtype=np.float64)
        if isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        else np.empty(0, dtype=np.float64)
    )
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        and qpos_cmd.shape == HAND_JOINT_SHAPE
        and np.all(np.isfinite(qpos_cmd))
        and limits_well_formed
        and np.all(qpos_cmd >= command_lower - 1e-12)
        and np.all(qpos_cmd <= command_upper + 1e-12)
        and np.all(qpos_cmd >= mechanical_lower - 1e-12)
        and np.all(qpos_cmd <= mechanical_upper + 1e-12)
    )
    if previous_qpos_cmd is not None or max_command_delta_rad is not None:
        previous = np.asarray(previous_qpos_cmd, dtype=np.float64)
        try:
            max_delta = np.broadcast_to(
                np.asarray(max_command_delta_rad, dtype=np.float64), HAND_JOINT_SHAPE
            )
        except (TypeError, ValueError):
            return False
        delta_well_formed = (
            previous.shape == HAND_JOINT_SHAPE
            and np.all(np.isfinite(previous))
            and np.all(np.isfinite(max_delta))
            and np.all(max_delta > 0.0)
            and np.all(np.abs(qpos_cmd - previous) <= max_delta + 1e-12)
        )
        well_formed = bool(well_formed and delta_well_formed)
    return bool(
        well_formed
        and _worker_command_is_current(
            command,
            expected_run_generation=expected_run_generation,
            now_monotonic_ns=now_monotonic_ns,
        )
    )


# ---------------------------------------------------------------------------
# Gate factory — wires SafetyGate to a planner's geometry callbacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSafetyGateConfig:
    """Configuration for :func:`planner_action_safety_gate`."""

    arm_joint_lower_rad: tuple[float, ...]
    arm_joint_upper_rad: tuple[float, ...]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]


def planner_action_safety_gate(
    config: ActionSafetyGateConfig,
    *,
    planner: Any,
) -> SafetyGate:
    """Build a :class:`SafetyGate` wired to the planner's workspace callback.

    Collision and transition geometry checks are **not** wired — they were
    removed from SafetyGate (2026-08-12).  Collision-free homing paths are
    planned independently through ``plan_joint_home_path`` and
    ``plan_band_alignment_path``, which call the collision model directly.
    """
    gate = SafetyGate(
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
    )
    gate.workspace_check = planner.is_workspace_segment_safe
    return gate


# ---------------------------------------------------------------------------
# Convenience publication — build an ActionCandidate, validate, and send
# ---------------------------------------------------------------------------


def build_action_candidate(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    *,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    now_ns: int | None = None,
    action_validity_s: float = 0.5,
) -> Any | None:
    """Build an ``ActionCandidate`` from raw joint targets.

    Allocates a fresh monotonic ``action_id`` from ``shared.arm_command_seq``
    and stamps the target/valid-until timestamps from
    ``shared.action_lead_time_s`` and ``action_validity_s``.  Returns ``None``
    when the optional observation anchor is non-positive or in the future.
    """
    from dexmani_real.policy.runtime import ActionCandidate

    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = int(time.monotonic_ns() if now_ns is None else now_ns)
    if observation_anchor_monotonic_ns is not None:
        anchor_ns = int(observation_anchor_monotonic_ns)
        if anchor_ns <= 0 or anchor_ns > now_ns:
            logger.warning(
                "build_action_candidate: action_id=%d rejected: invalid observation anchor",
                action_id,
            )
            return None
    return ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        run_generation=int(shared.run_generation.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + int(float(shared.action_lead_time_s) * 1e9),
        valid_until_monotonic_ns=now_ns + int(float(action_validity_s) * 1e9),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
        hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
        is_hold=is_hold,
    )


def validate_and_send_candidate(
    shared: Any,
    candidate: Any,  # ActionCandidate
    *,
    gate: SafetyGate,
    prepare_timeout_s: float = 0.06,
    current_arm_qpos: np.ndarray | None = None,
    current_hand_qpos: np.ndarray | None = None,
    dt_s: float | None = None,
) -> Any | None:
    """Validate a pre-built candidate through the gate and publish it.

    Reads current arm/hand feedback when the caller does not supply it, runs
    :meth:`SafetyGate.validate`, and publishes via :func:`send_command`.  This is
    the publication tail shared by VR teleop, keyboard/replay, and the
    learned-policy coordinator.

    The coupled-hand mechanical/delta preflight is deliberately absent: its delta
    reference and rejection policy are path-dependent (last-published vs
    last-accepted command), so each caller runs it before calling this function.

    Returns:
        The accepted ``ActionCandidate`` on publication, or ``None`` when the
        gate rejects or the transport fails.
    """
    action_id = int(candidate.action_id)

    if current_arm_qpos is None:
        arm_result = shared.arm_state_ring.read_latest()
        if arm_result is None:
            logger.warning("validate_and_send_candidate: arm feedback unavailable")
            return None
        arm_record = arm_result[0][0]
        if not bool(arm_record["connected"]) or not bool(arm_record["state_valid"]):
            logger.warning("validate_and_send_candidate: arm feedback unhealthy")
            return None
        current_arm = np.asarray(arm_record["qpos"], dtype=np.float64)
        if current_arm.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current_arm)):
            return None
    else:
        current_arm = np.asarray(current_arm_qpos, dtype=np.float64)

    if current_hand_qpos is None:
        current_hand = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is not None:
            hand_record = hand_result[0][0]
            if bool(hand_record["connected"]) and bool(hand_record["state_valid"]):
                current_hand = np.asarray(hand_record["qpos"], dtype=np.float64)
        elif candidate.hand_qpos is not None:
            logger.warning("validate_and_send_candidate: hand feedback unavailable")
            return None
    else:
        current_hand = np.asarray(current_hand_qpos, dtype=np.float64)

    ctrl_dt = (
        1.0 / float(getattr(shared, "action_control_hz", 16.0))
        if dt_s is None
        else float(dt_s)
    )
    gate_result = gate.validate(
        candidate,
        current_arm_qpos=current_arm,
        current_hand_qpos=current_hand,
        dt_s=ctrl_dt,
        run_generation=int(shared.run_generation.value),
    )
    if not gate_result.accepted or gate_result.candidate is None:
        logger.warning(
            "validate_and_send_candidate: action_id=%d rejected by safety gate: %s",
            action_id,
            gate_result.reason or "unspecified",
        )
        return None

    if not send_command(shared, gate_result.candidate, prepare_timeout_s=prepare_timeout_s):
        return None
    return gate_result.candidate


def publish_joint_targets(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    is_hold: bool = False,
    prepare_timeout_s: float = 0.05,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    dt_s: float | None = None,
    safety_gate: SafetyGate | None = None,
    wait_applied: bool = False,
    apply_timeout_s: float = 0.5,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    hand_max_delta_rad: float | np.ndarray | None = None,
) -> Any | None:
    """Validate a joint-space target through the gate and publish it.

    This is a convenience wrapper used by keyboard teleop, calibration, and replay — it
    builds an ``ActionCandidate`` from raw joint arrays, runs
    the full validation pipeline, and calls :func:`send_command`.  When the
    candidate carries a hand target, a coupled-hand preflight
    (:func:`validate_hand_command_delta`) additionally rejects-whole the rated
    mechanical envelope and the command-to-command delta *before* the arm
    endpoint is enqueued, so a rejected hand command cannot desync the arm from
    the hand.

    ``hand_mechanical_lower_rad`` / ``hand_mechanical_upper_rad`` default to the
    rated device envelope; ``hand_max_delta_rad`` is optional (``None`` skips
    the command-to-command delta check while still enforcing operational +
    mechanical bounds).

    Returns:
        The accepted ``ActionCandidate`` that was published (and, when
        ``wait_applied`` is true, acknowledged by arm feedback — plus hand
        feedback when the candidate carries a hand target), or ``None``
        when validation, publication, or acknowledgement failed.
    """
    if safety_gate is None:
        logger.error("publish_joint_targets: no safety gate configured")
        return None
    gate = safety_gate

    candidate = build_action_candidate(
        shared,
        arm_qpos,
        hand_qpos,
        is_hold=is_hold,
        observation_id=observation_id,
        observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
    )
    if candidate is None:
        return None
    action_id = int(candidate.action_id)

    # Read current arm feedback
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        logger.warning(
            "publish_joint_targets: action_id=%d rejected: arm state ring is empty",
            action_id,
        )
        return None
    arm_record = arm_result[0][0]
    if not bool(arm_record["connected"]) or not bool(arm_record["state_valid"]):
        logger.warning(
            "publish_joint_targets: action_id=%d rejected: arm feedback invalid "
            "(connected=%s state_valid=%s)",
            action_id,
            bool(arm_record["connected"]),
            bool(arm_record["state_valid"]),
        )
        return None
    # ``arm_record`` is a scalar structured record: its qpos field is already
    # the full 7-DoF vector.  Indexing it again selects joint 0 and turns the
    # safety gate input into a scalar.
    current_arm = np.asarray(arm_record["qpos"], dtype=np.float64)

    current_hand = np.zeros(12, dtype=np.float64)
    previous_hand_cmd: np.ndarray | None = None
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_record = hand_result[0][0]
        if bool(hand_record["connected"]) and bool(hand_record["state_valid"]):
            current_hand = np.asarray(hand_record["qpos"], dtype=np.float64)
            # The command-to-command delta reference is the last *accepted
            # command*, not measured feedback (which may legitimately lag).
            last_cmd = np.asarray(hand_record["last_cmd_qpos"], dtype=np.float64)
            if last_cmd.shape == HAND_JOINT_SHAPE and np.all(np.isfinite(last_cmd)):
                previous_hand_cmd = last_cmd
    elif hand_qpos is not None:
        logger.warning(
            "publish_joint_targets: action_id=%d rejected: hand state ring is empty",
            action_id,
        )
        return None

    # Coupled hand preflight: the gate only enforces operational limits.  Check
    # the rated mechanical envelope and the command-to-command delta here
    # (reject-whole, never clip) before the arm endpoint is enqueued, so a
    # rejected hand command cannot desync the arm from the hand.  The gate does
    # not mutate the candidate, so running this before the gate/send tail is
    # outcome-identical to the historical after-gate order.
    if candidate.hand_qpos is not None:
        mechanical_lower = (
            np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
            if hand_mechanical_lower_rad is None
            else np.asarray(hand_mechanical_lower_rad, dtype=np.float64)
        )
        mechanical_upper = (
            np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
            if hand_mechanical_upper_rad is None
            else np.asarray(hand_mechanical_upper_rad, dtype=np.float64)
        )
        try:
            validate_hand_command_delta(
                candidate.hand_qpos,
                previous_hand_cmd,
                gate.hand_low,
                gate.hand_high,
                mechanical_lower,
                mechanical_upper,
                hand_max_delta_rad,
            )
        except ValueError as exc:
            logger.warning(
                "publish_joint_targets: action_id=%d hand command rejected by "
                "mechanical/delta preflight: %s",
                action_id,
                exc,
            )
            return None

    published = validate_and_send_candidate(
        shared,
        candidate,
        gate=gate,
        prepare_timeout_s=prepare_timeout_s,
        current_arm_qpos=current_arm,
        current_hand_qpos=current_hand,
        dt_s=dt_s,
    )
    if published is None:
        return None
    if wait_applied:
        if not (np.isfinite(apply_timeout_s) and apply_timeout_s > 0):
            raise ValueError("apply_timeout_s must be finite and positive")
        deadline_s = time.monotonic() + float(apply_timeout_s)
        # A candidate carrying a hand_qpos is published to both actuators under
        # the same action_id (see send_command), so a synchronous caller must
        # observe BOTH applied before returning.  Arm-only candidates keep the
        # existing arm last_cmd_seq >= action_id gate.
        with_hand = published.hand_qpos is not None
        while time.monotonic() < deadline_s:
            if bool(shared.error_state.value) or not bool(shared.is_running.value):
                break
            latest = shared.arm_state_ring.read_latest()
            arm_ok = latest is not None and int(latest[0][0]["last_cmd_seq"]) >= action_id
            if not with_hand:
                if arm_ok:
                    return published
            else:
                hand_latest = shared.hand_state_ring.read_latest()
                if hand_latest is not None:
                    hand_seq = int(hand_latest[0][0]["last_cmd_seq"])
                    if hand_seq > action_id:
                        logger.warning(
                            "publish_joint_targets: action_id=%d was superseded by hand action_id=%d",
                            action_id,
                            hand_seq,
                        )
                        return None
                    if arm_ok and hand_seq == action_id:
                        hs = hand_latest[0][0]
                        healthy = (
                            bool(hs["connected"])
                            and bool(hs["state_valid"])
                            and not bool(hs["error_state"])
                            and bool(hs["send_healthy"])
                            and bool(hs["read_healthy"])
                        )
                        if healthy:
                            return published
            time.sleep(0.005)
        logger.warning(
            "publish_joint_targets: action_id=%d was not acknowledged within %.3fs",
            action_id,
            apply_timeout_s,
        )
        return None
    return published


# ---------------------------------------------------------------------------
# Hand command bound/delta preflight (shared by every coupled publish path)
# ---------------------------------------------------------------------------


def validate_hand_command_delta(
    hand_cmd: np.ndarray,
    previous: np.ndarray | None,
    operational_lower: np.ndarray,
    operational_upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
    max_delta_rad: float | np.ndarray | None,
) -> np.ndarray:
    """Validate one hand target against operational + rated mechanical bounds
    and (optionally) a command-to-command delta; reject-whole, never clip.

    Shared preflight for every coupled hand path (teleop, replay, return-home).
    ``previous`` is the reference *command* for the delta bound and is always a
    command, never measured feedback, so contact and torque-limited steady-state
    lag are valid outcomes and must not reject a valid next command.  Which
    command is the reference is path-dependent: replay/keyboard/calibrate pass
    the worker's last *accepted* command (``last_cmd_qpos``), while VR teleop
    passes the last *published* command (``ctx.prev_hand_qpos``).  The worker's
    authoritative delta check remains the final backstop on either path.  Raises
    ``ValueError`` on any violation and returns a copy otherwise.
    """
    command = np.asarray(hand_cmd, dtype=np.float64)
    op_lower = np.asarray(operational_lower, dtype=np.float64)
    op_upper = np.asarray(operational_upper, dtype=np.float64)
    mech_lower = np.asarray(mechanical_lower, dtype=np.float64)
    mech_upper = np.asarray(mechanical_upper, dtype=np.float64)
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)

    if command.shape != HAND_JOINT_SHAPE:
        raise ValueError(
            f"hand command must have shape {HAND_JOINT_SHAPE}, got {command.shape}"
        )
    for label, value in (
        ("operational lower", op_lower),
        ("operational upper", op_upper),
        ("mechanical lower", mech_lower),
        ("mechanical upper", mech_upper),
    ):
        if value.shape != HAND_JOINT_SHAPE:
            raise ValueError(f"hand {label} limits must have shape {HAND_JOINT_SHAPE}")
    if not np.all(
        np.isfinite(np.concatenate((command, op_lower, op_upper, mech_lower, mech_upper)))
    ):
        raise ValueError("hand command and limit arrays must be finite")
    if np.any(op_lower > op_upper) or np.any(mech_lower > mech_upper):
        raise ValueError("hand operational and mechanical limits must be ordered")
    if np.any(mech_lower < rated_lower) or np.any(mech_upper > rated_upper):
        raise ValueError("hand mechanical limits cannot exceed the rated device envelope")
    if np.any(op_lower < mech_lower) or np.any(op_upper > mech_upper):
        raise ValueError("hand operational limits must be inside mechanical limits")
    if np.any(command < op_lower - 1e-12) or np.any(command > op_upper + 1e-12):
        raise ValueError("hand command violates operational joint limits")
    if np.any(command < mech_lower - 1e-12) or np.any(command > mech_upper + 1e-12):
        raise ValueError("hand command violates rated mechanical joint limits")

    if max_delta_rad is not None:
        if previous is None:
            raise ValueError(
                "hand command delta check requires a previous accepted command"
            )
        prev = np.asarray(previous, dtype=np.float64)
        try:
            max_delta = np.broadcast_to(
                np.asarray(max_delta_rad, dtype=np.float64), HAND_JOINT_SHAPE
            )
        except (TypeError, ValueError):
            raise ValueError("hand max command delta must broadcast to twelve values") from None
        if prev.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(prev)):
            raise ValueError("hand previous command must be a finite 12-vector")
        if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
            raise ValueError("hand max command delta must be finite and positive")
        if np.any(np.abs(command - prev) > max_delta + 1e-12):
            raise ValueError("hand command violates command-to-command delta limit")

    return command.copy()


# ---------------------------------------------------------------------------
# Hand homing utility (command acceptance only; no execution convergence gate)
# ---------------------------------------------------------------------------


def publish_hand_home_and_wait_applied(
    shared: Any,
    home_qpos: np.ndarray,
    *,
    command_lower_rad: np.ndarray,
    command_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    max_command_delta_rad: float | np.ndarray | None = None,
    timeout_s: float = 1.0,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    abort_requested: Any = None,
) -> bool:
    """Publish exact hand-home and wait only for worker/SDK acceptance.

    The configured endpoint must lie inside both the operational command box
    and the rated mechanical box. When a command-delta limit is configured,
    this function publishes explicit linear milestones from the worker's last
    accepted command; it never clips a candidate. Success means every SDK send,
    including the exact final home endpoint, was acknowledged. Measured qpos is
    deliberately not compared with the target because contact and steady-state
    position error are valid.
    """
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(
            "hand home command acknowledgement timeout must be finite and positive"
        )
    # Bound validation is shared with the other coupled hand paths; a
    # violation raises (reject-whole, never clip).  Delta is not checked here:
    # the milestone loop below enforces the command-to-command bound explicitly.
    target = validate_hand_command_delta(
        home_qpos,
        None,
        command_lower_rad,
        command_upper_rad,
        mechanical_lower_rad,
        mechanical_upper_rad,
        max_delta_rad=None,
    )
    command_lower = np.asarray(command_lower_rad, dtype=np.float64)
    command_upper = np.asarray(command_upper_rad, dtype=np.float64)
    deadline_s = time.monotonic() + timeout_s
    initial_result = shared.hand_state_ring.read_latest()
    if initial_result is None:
        logger.warning("hand home rejected: hand state ring is empty")
        return False
    initial_state = initial_result[0][0]
    start = np.asarray(initial_state["last_cmd_qpos"], dtype=np.float64)
    initial_healthy = (
        start.shape == HAND_JOINT_SHAPE
        and np.all(np.isfinite(start))
        and bool(initial_state["connected"])
        and bool(initial_state["state_valid"])
        and not bool(initial_state["error_state"])
        and bool(initial_state["send_healthy"])
        and bool(initial_state["read_healthy"])
    )
    if not initial_healthy:
        logger.warning(
            "hand home rejected: last accepted hand command or worker health is invalid"
        )
        return False
    if np.any(start < command_lower - 1e-12) or np.any(start > command_upper + 1e-12):
        logger.warning(
            "hand home rejected: last accepted hand command violates operational limits"
        )
        return False

    if max_command_delta_rad is None:
        max_delta = None
        milestone_count = 1
    else:
        max_delta = np.broadcast_to(
            np.asarray(max_command_delta_rad, dtype=np.float64), HAND_JOINT_SHAPE
        )
        if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
            raise ValueError(
                "hand max command delta must broadcast to twelve finite positive values"
            )
        milestone_count = max(
            1, int(np.ceil(float(np.max(np.abs(target - start) / max_delta))))
        )

    last_action_id = 0
    milestone_index = 0
    acknowledged = False
    previous = start
    for milestone_index in range(1, milestone_count + 1):
        if time.monotonic() >= deadline_s:
            break
        alpha = milestone_index / milestone_count
        milestone = (
            target.copy()
            if milestone_index == milestone_count
            else start + alpha * (target - start)
        )
        if max_delta is not None and np.any(
            np.abs(milestone - previous) > max_delta + 1e-12
        ):
            raise RuntimeError(
                "generated hand-home milestone violates configured command delta"
            )
        previous = milestone

        with shared.arm_command_seq.get_lock():
            action_id = int(shared.arm_command_seq.value) + 1
            shared.arm_command_seq.value = action_id
        last_action_id = action_id
        now_ns = time.monotonic_ns()
        frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
        frame["run_generation"][0] = int(shared.run_generation.value)
        frame["observation_id"][0] = action_id
        frame["action_id"][0] = action_id
        frame["created_monotonic_ns"][0] = now_ns
        frame["target_monotonic_ns"][0] = now_ns
        frame["valid_until_monotonic_ns"][0] = now_ns + int(
            max(0.3, deadline_s - time.monotonic() + 0.1) * 1e9
        )
        frame["is_hold"][0] = 0
        frame["qpos_cmd"][0] = milestone
        shared.hand_cmd_ring.write(frame)

        acknowledged = False
        while time.monotonic() < deadline_s:
            if abort_requested is not None and abort_requested():
                return False
            if bool(getattr(getattr(shared, "estop_request", None), "value", False)):
                return False
            if bool(shared.error_state.value):
                return False
            if check_is_running and not bool(shared.is_running.value):
                return False
            if heartbeat:
                shared.set_heartbeat("policy", time.monotonic())

            result = shared.hand_state_ring.read_latest()
            if result is not None:
                state = result[0][0]
                applied_id = int(state["last_cmd_seq"])
                if applied_id > action_id:
                    logger.warning(
                        "hand home action_id=%d was superseded by action_id=%d before acknowledgement",
                        action_id,
                        applied_id,
                    )
                    return False
                if applied_id == action_id:
                    healthy = (
                        bool(state["connected"])
                        and bool(state["state_valid"])
                        and not bool(state["error_state"])
                        and bool(state["send_healthy"])
                        and bool(state["read_healthy"])
                    )
                    if healthy:
                        acknowledged = True
                        break
            time.sleep(0.01)
        if not acknowledged:
            break

    if last_action_id and milestone_index == milestone_count and acknowledged:
        if verbose:
            print(
                f"  hand: home command accepted (action_id={last_action_id}, milestones={milestone_count})",
                flush=True,
            )
        return True

    logger.warning(
        "hand home action_id=%d was not acknowledged within %.3fs",
        last_action_id,
        timeout_s,
    )
    return False

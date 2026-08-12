"""Unified safety gate — the single validation boundary for all action paths.

Every coordinator (teleop, learned policy, keyboard, replay, calibration) must
route candidates through :class:`SafetyGate` before :func:`send_command` writes
to the actuator IPC primitives.  Workers trust the gate and apply commands
immediately with only hardware-level checks (safety state, dtype, finite, SDK
return code).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import ARM_COMMAND_DTYPE, ARM_JOINT_SHAPE, HAND_COMMAND_DTYPE, HAND_JOINT_SHAPE

logger = get_logger(__name__)


def advance_policy_epoch(shared: Any) -> int:
    """Invalidate commands prepared under a previous policy epoch."""
    lock_getter = getattr(shared.policy_epoch, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            shared.policy_epoch.value = int(shared.policy_epoch.value) + 1
            return int(shared.policy_epoch.value)
    shared.policy_epoch.value = int(shared.policy_epoch.value) + 1
    return int(shared.policy_epoch.value)


@dataclass(frozen=True)
class GateResult:
    """Outcome of :meth:`SafetyGate.validate`."""

    accepted: bool
    candidate: Any  # ActionCandidate (lazy import to avoid cycle)
    reason: str = ""
    delta_clamped: bool = False


class SafetyGate:
    """Single validation boundary for all action paths.

    Pipeline (short-circuit, fail-closed):

    1. **Well-formed** — representation, shapes, finite values
    2. **Joint limits** — commanded actuators only; hold actuators skip
    3. **Velocity clamp** — per-joint clip to ``[current ± max_vel·dt]``
    4. **Collision** — ``collision_model.check_collision(clamped_qpos)``
    5. **Workspace** — optional segment check

    Workers apply the clamped output immediately; they do **not** re-validate
    epochs, sessions, action IDs, or temporal windows — the velocity clamp is
    the universal backstop for stale or corrupt commands.
    """

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
        arm_max_velocity_rad_s: float,
        hand_max_velocity_rad_s: float,
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
        if not np.all(np.isfinite(concat)) or np.any(_arm_low > _arm_high) or np.any(_hand_low > _hand_high):
            raise ValueError("joint limits must be finite and ordered")
        if not (np.isfinite(arm_max_velocity_rad_s) and arm_max_velocity_rad_s > 0):
            raise ValueError("arm_max_velocity_rad_s must be finite and positive")
        if not (np.isfinite(hand_max_velocity_rad_s) and hand_max_velocity_rad_s > 0):
            raise ValueError("hand_max_velocity_rad_s must be finite and positive")

        self.arm_low = _arm_low
        self.arm_high = _arm_high
        self.hand_low = _hand_low
        self.hand_high = _hand_high
        self.arm_max_vel = arm_max_velocity_rad_s
        self.hand_max_vel = hand_max_velocity_rad_s

    # -- callbacks set after construction (avoids circular imports) ---------
    collision_check: Any = None  # Callable[[np.ndarray], bool] | None
    workspace_check: Any = None  # Callable[[np.ndarray, np.ndarray], bool] | None

    def validate(
        self,
        candidate: Any,  # ActionCandidate
        *,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
        dt_s: float,
        session_generation: int,
    ) -> GateResult:
        """Run the full validation pipeline.

        Args:
            candidate: The proposed ``ActionCandidate``.
            current_arm_qpos: Latest measured arm joint positions [rad].
            current_hand_qpos: Latest measured hand joint positions [rad].
            dt_s: Action period (``1 / control_hz``).
            session_generation: Expected session generation.

        Returns:
            ``GateResult`` with the (possibly clamped) safe candidate.
        """
        # 1 ── Well-formed ────────────────────────────────────────────
        if (
            candidate.representation != "joint_position"
            or candidate.units != "rad"
            or candidate.frame != "robot_joint"
        ):
            return GateResult(False, candidate, "unsupported representation/units/frame")

        if candidate.session_generation != session_generation:
            return GateResult(False, candidate, "session generation mismatch")

        arm_start = np.asarray(current_arm_qpos, dtype=np.float64)
        hand_start = np.asarray(current_hand_qpos, dtype=np.float64)
        if arm_start.shape != ARM_JOINT_SHAPE or hand_start.shape != HAND_JOINT_SHAPE:
            return GateResult(False, candidate, "invalid current joint state shape")
        if not np.all(np.isfinite(arm_start)) or not np.all(np.isfinite(hand_start)):
            return GateResult(False, candidate, "current joint state contains NaN/Inf")

        arm_end = arm_start.copy() if candidate.arm_qpos is None else np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        hand_end = hand_start.copy() if candidate.hand_qpos is None else np.asarray(candidate.hand_qpos, dtype=np.float64).copy()
        if arm_end.shape != ARM_JOINT_SHAPE or hand_end.shape != HAND_JOINT_SHAPE:
            return GateResult(False, candidate, "invalid candidate joint shape")
        if not np.all(np.isfinite(arm_end)) or not np.all(np.isfinite(hand_end)):
            return GateResult(False, candidate, "candidate contains NaN/Inf")

        # 2 ── Joint limits (commanded actuators only) ─────────────────
        if candidate.arm_qpos is not None and (np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)):
            return GateResult(False, candidate, "arm joint limit violation")
        if candidate.hand_qpos is not None and (np.any(hand_end < self.hand_low) or np.any(hand_end > self.hand_high)):
            return GateResult(False, candidate, "hand joint limit violation")

        # 3 ── Velocity clamp ──────────────────────────────────────────
        if not (np.isfinite(dt_s) and dt_s > 0):
            return GateResult(False, candidate, "invalid dt")

        arm_delta = self.arm_max_vel * dt_s
        hand_delta = self.hand_max_vel * dt_s
        safe_arm = np.clip(arm_end, arm_start - arm_delta, arm_start + arm_delta)
        safe_hand = np.clip(hand_end, hand_start - hand_delta, hand_start + hand_delta)
        delta_clamped = not np.array_equal(safe_arm, arm_end) or not np.array_equal(safe_hand, hand_end)

        # 4 ── Collision ───────────────────────────────────────────────
        if self.collision_check is not None:
            try:
                if candidate.arm_qpos is not None and candidate.hand_qpos is not None:
                    # 19-DOF combined
                    full_qpos = np.concatenate([safe_arm, safe_hand])
                elif candidate.arm_qpos is not None:
                    full_qpos = safe_arm
                else:
                    full_qpos = safe_hand  # hand-only (rare)
                if self.collision_check(full_qpos):
                    return GateResult(False, candidate, "collision", delta_clamped=delta_clamped)
            except Exception:
                logger.warning("SafetyGate: collision check failed closed", exc_info=True)
                return GateResult(False, candidate, "collision check failed")

        # 5 ── Workspace (optional) ────────────────────────────────────
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, safe_arm):
                    return GateResult(False, candidate, "workspace", delta_clamped=delta_clamped)
            except Exception:
                logger.warning("SafetyGate: workspace check failed closed", exc_info=True)
                return GateResult(False, candidate, "workspace check failed")

        # Re-assemble with clamped values
        from dataclasses import replace

        try:
            safe_candidate = replace(
                candidate,
                arm_qpos=safe_arm if candidate.arm_qpos is not None else None,
                hand_qpos=safe_hand if candidate.hand_qpos is not None else None,
            )
        except TypeError:
            # Non-dataclass candidate (e.g., Mock in tests) — work with what we have
            safe_candidate = candidate
            if candidate.arm_qpos is not None:
                object.__setattr__(candidate, 'arm_qpos', safe_arm)
            if candidate.hand_qpos is not None:
                object.__setattr__(candidate, 'hand_qpos', safe_hand)
        return GateResult(True, safe_candidate, delta_clamped=delta_clamped)


# ---------------------------------------------------------------------------
# Command serialization and publication
# ---------------------------------------------------------------------------


def _make_arm_command(candidate: Any, now_monotonic_ns: int, target_monotonic_ns: int) -> np.ndarray:
    """Serialize an ActionCandidate into an ARM_COMMAND_DTYPE record."""
    if candidate.arm_qpos is None:
        raise ValueError("candidate has no arm command")
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["session_generation"][0] = candidate.session_generation
    frame["policy_epoch"][0] = candidate.policy_epoch
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["chunk_id"][0] = candidate.chunk_id
    frame["step_index"][0] = candidate.step_index
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = target_monotonic_ns + int(3e8)  # +300ms
    frame["is_hold"][0] = int(candidate.arm_qpos is None)
    frame["qpos_cmd"][0] = candidate.arm_qpos
    return frame


def _make_hand_command(candidate: Any, now_monotonic_ns: int, target_monotonic_ns: int) -> np.ndarray:
    """Serialize an ActionCandidate into a HAND_COMMAND_DTYPE record."""
    if candidate.hand_qpos is None:
        raise ValueError("candidate has no hand command")
    frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
    frame["session_generation"][0] = candidate.session_generation
    frame["policy_epoch"][0] = candidate.policy_epoch
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["chunk_id"][0] = candidate.chunk_id
    frame["step_index"][0] = candidate.step_index
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = target_monotonic_ns + int(3e8)
    frame["is_hold"][0] = int(candidate.hand_qpos is None)
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
        hold) or when the candidate's temporal window has already closed.
    """
    timeout = prepare_timeout_s if prepare_timeout_s is not None else policy_defaults.action_prepare_timeout_s
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("prepare_timeout_s must be finite and positive")

    now_ns = time.monotonic_ns()
    lead_time_s = float(getattr(shared, "action_lead_time_s", 0.05))
    target_ns = now_ns + int(lead_time_s * 1e9)

    # Reject when the target time has already passed or the validity window
    # is behind wall-clock (can happen after a long gate evaluation).
    if target_ns <= now_ns or (candidate.valid_until_monotonic_ns and candidate.valid_until_monotonic_ns < now_ns):
        logger.error("send_command: action_id=%d temporal window closed", candidate.action_id)
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
            logger.error("send_command: arm queue full (action_id=%d)", candidate.action_id)
            return False

    if candidate.hand_qpos is not None:
        hand_frame = _make_hand_command(candidate, now_ns, target_ns)
        shared.hand_cmd_ring.write(hand_frame)

    return True


# ---------------------------------------------------------------------------
# Worker-side validation (minimal — trust the gate)
# ---------------------------------------------------------------------------


def worker_validate_arm(command: np.ndarray) -> bool:
    """Minimal hardware-level check for an arm command from the queue.

    Returns True when the command is well-formed enough to pass to the SDK.
    The gate already validated limits, velocity, and collision.
    """
    return (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == ARM_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    )


def worker_validate_hand(command: np.ndarray) -> bool:
    """Minimal hardware-level check for a hand command from the ring.

    Returns True when the command is well-formed enough to pass to the SDK.
    The gate already validated limits, velocity, and collision.
    """
    return (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    )


# ---------------------------------------------------------------------------
# Gate factory — wires SafetyGate to a planner's geometry callbacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSafetyGateConfig:
    """Configuration for :func:`planner_action_safety_gate`.

    This dataclass provides a typed configuration for
    :func:`planner_action_safety_gate`.
    """

    arm_joint_lower_rad: tuple[float, ...]
    arm_joint_upper_rad: tuple[float, ...]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]
    arm_max_velocity_rad_s: float
    hand_max_velocity_rad_s: float
    observation_max_age_s: float = 0.25
    require_geometry_checks: bool = True


def planner_action_safety_gate(
    config: ActionSafetyGateConfig,
    *,
    planner: Any,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    transition_step_rad: float = 0.02,
    enable_table_check: bool = True,
) -> SafetyGate:
    """Build a geometry-aware :class:`SafetyGate`, wired to planner callbacks."""
    gate = SafetyGate(
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
        arm_max_velocity_rad_s=config.arm_max_velocity_rad_s,
        hand_max_velocity_rad_s=config.hand_max_velocity_rad_s,
    )
    if config.require_geometry_checks:
        gate.collision_check = planner.collision_model.check_collision
        gate.workspace_check = planner.is_workspace_segment_safe
    return gate


# ---------------------------------------------------------------------------
# Convenience publication — build an ActionCandidate, validate, and send
# ---------------------------------------------------------------------------


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
) -> Any | None:
    """Validate a joint-space target through the gate and publish it.

    This is a convenience wrapper used by keyboard teleop, calibration, and
    replay — it builds an ``ActionCandidate`` from raw joint arrays, runs
    the full validation pipeline, and calls :func:`send_command`.

    Returns:
        The (possibly clamped) ``ActionCandidate`` that was published,
        or ``None`` when validation or publication failed.
    """
    from dexmani_real.policy.runtime import ActionCandidate

    if safety_gate is None:
        return None
    gate = safety_gate

    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = time.monotonic_ns()
    lead_time_s = float(getattr(shared, "action_lead_time_s", 0.05))

    candidate = ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        session_generation=int(shared.session_generation.value),
        policy_epoch=int(shared.policy_epoch.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + int(lead_time_s * 1e9),
        valid_until_monotonic_ns=now_ns + int(0.5 * 1e9),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
        hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
        chunk_id=action_id,
        step_index=0,
        is_hold=is_hold,
    )

    # Read current arm feedback
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        return None
    arm_record = arm_result[0][0]
    if not bool(arm_record["connected"]) or not bool(arm_record["state_valid"]):
        return None
    current_arm = np.asarray(arm_record["qpos"][0], dtype=np.float64)

    current_hand = np.zeros(12, dtype=np.float64)
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_record = hand_result[0][0]
        if bool(hand_record["connected"]) and bool(hand_record["state_valid"]):
            current_hand = np.asarray(hand_record["qpos"][0], dtype=np.float64)
    elif hand_qpos is not None:
        return None

    ctrl_dt = 1.0 / float(getattr(shared, "action_control_hz", 16.0)) if dt_s is None else float(dt_s)
    gate_result = gate.validate(
        candidate,
        current_arm_qpos=current_arm,
        current_hand_qpos=current_hand,
        dt_s=ctrl_dt,
        session_generation=int(shared.session_generation.value),
    )
    if not gate_result.accepted or gate_result.candidate is None:
        return None

    if not send_command(shared, gate_result.candidate, prepare_timeout_s=prepare_timeout_s):
        return None
    return gate_result.candidate


# ---------------------------------------------------------------------------
# Hand homing utility (writes directly to hand_cmd_ring, polls for convergence)
# ---------------------------------------------------------------------------


def hand_home_converge(
    shared: Any,
    home_qpos: np.ndarray,
    *,
    timeout_s: float = 5.0,
    tol_rad: float = 0.05,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    safety_gate: SafetyGate | None = None,
    abort_requested: Any = None,
) -> tuple[bool, np.ndarray | None]:
    """Drive the hand to *home_qpos* and wait for measured convergence.

    Writes directly to ``hand_cmd_ring`` (latest-wins) and polls
    ``hand_state_ring`` until the max joint error is below *tol_rad* or
    *timeout_s* expires.

    Returns:
        ``(True, final_qpos)`` on success, ``(False, None)`` otherwise.
    """
    deadline = time.monotonic() + timeout_s
    first = True
    while time.monotonic() < deadline:
        if abort_requested is not None and abort_requested():
            return False, None
        if check_is_running and not shared.is_running.value:
            break
        if heartbeat:
            shared.policy_heartbeat_s.value = time.monotonic()

        frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
        frame["qpos_cmd"][0] = home_qpos
        shared.hand_cmd_ring.write(frame)

        hand_result = shared.hand_state_ring.read_latest()
        if hand_result is not None:
            state = hand_result[0][0]
            try:
                if bool(state["connected"]) and bool(state["state_valid"]) and not bool(state["error_state"]):
                    current = np.asarray(state["qpos"], dtype=np.float64)
                    if current.shape == (12,) and np.all(np.isfinite(current)):
                        err = float(np.max(np.abs(current - home_qpos)))
                        if err < tol_rad:
                            return True, current.copy()
                        if verbose and first:
                            print(f"  hand: homing... (max_err={np.rad2deg(err):.0f}°)", flush=True)
                            first = False
            except Exception:
                pass
        time.sleep(0.1)
    return False, None

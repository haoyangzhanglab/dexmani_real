"""Collision-checked arm homing shared by experiment entry points.

The planner densely validates a joint-space path (self/table/environment
collision, joint limits) and returns a typed already-home/safe/unsafe result.
Only a safe result supplies sparse milestones.  The requester queues
``(waypoints, final_qpos)`` to the arm worker, which drives them as a
blocking ``XArm7.home()`` in Mode 0; completion is observed from the arm
state ring (a fresh Mode-6 frame at the canonical home), not from an RPC
acknowledgement.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from queue import Full
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.defaults import arm
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning.paths import (
    HomePathCandidate,
    HomePathStatus,
    compute_band_alignment_path,
    compute_joint_home_path,
)
from dexmani_real.robot_spec import ARM_JOINT_SHAPE
from dexmani_real.runtime.safety import SafetyState, revoke_motion
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig
    from dexmani_real.planning import XArm7MotionPlanner

logger = get_logger(__name__)

_HOME_RESULT_POLL_S = 0.1
_ARM_HEARTBEAT_MAX_AGE_S = 1.0
_MIN_HOME_TIMEOUT_S = 10.0
_HOME_TIMEOUT_PADDING_S = 5.0
_RESULT_TIMEOUT_PADDING_S = 2.0


class ArmHomeStatus(str, Enum):
    """Stable outcomes for one collision-checked arm-home request."""

    REACHED = "reached"
    ESTOP_REQUESTED = "estop_requested"
    RUNTIME_STOPPED = "runtime_stopped"
    FAULTED = "faulted"
    NOT_ARMED = "not_armed"
    INVALID_TARGET = "invalid_target"
    INVALID_CURRENT_STATE = "invalid_current_state"
    PREHOME_STATE_UNAVAILABLE = "prehome_state_unavailable"
    PLANNER_UNAVAILABLE = "planner_unavailable"
    PLANNING_FAILED = "planning_failed"
    NO_SAFE_PATH = "no_safe_path"
    GENERATION_CHANGED = "generation_changed"
    QUEUE_FULL = "queue_full"
    QUEUE_ERROR = "queue_error"
    COMPLETION_TIMEOUT = "completion_timeout"


@dataclass(frozen=True)
class ArmHomeResult:
    """Typed result preserving the reason a home request did not complete."""

    status: ArmHomeStatus
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is ArmHomeStatus.REACHED


@dataclass(frozen=True)
class ArmHomeConfig:
    """Timing and convergence policy for one arm-home workflow."""

    request_queue_timeout_s: float = arm.homing.request_queue_timeout_s
    prehome_timeout_s: float = arm.homing.convergence_timeout_s
    state_max_age_s: float = arm.homing.state_max_age_s
    max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s)
    target_timeout_s: float = arm.homing.target_timeout_s
    arm_heartbeat_max_age_s: float = _ARM_HEARTBEAT_MAX_AGE_S
    stationary_velocity_rad_s: float = arm.homing.velocity_convergence_rad_s
    result_tolerance_rad: float = arm.homing.convergence_rad
    publish_policy_heartbeat: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "request_queue_timeout_s",
            "prehome_timeout_s",
            "state_max_age_s",
            "max_speed_rad_s",
            "target_timeout_s",
            "arm_heartbeat_max_age_s",
            "stationary_velocity_rad_s",
            "result_tolerance_rad",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")

    @classmethod
    def from_runtime(
        cls,
        runtime: "ResolvedRuntimeConfig",
        *,
        publish_policy_heartbeat: bool = True,
    ) -> "ArmHomeConfig":
        """Project canonical runtime values into the homing-owned contract."""
        return cls(
            request_queue_timeout_s=float(runtime.arm.homing.request_queue_timeout_s),
            prehome_timeout_s=float(runtime.arm.homing.convergence_timeout_s),
            state_max_age_s=float(runtime.arm.homing.state_max_age_s),
            max_speed_rad_s=float(np.deg2rad(runtime.arm.homing.max_speed_deg_s)),
            target_timeout_s=float(runtime.arm.homing.target_timeout_s),
            arm_heartbeat_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
            stationary_velocity_rad_s=float(
                runtime.arm.homing.velocity_convergence_rad_s
            ),
            result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
            publish_policy_heartbeat=publish_policy_heartbeat,
        )


def _describe_band_diff(wrapped: np.ndarray, canonical: np.ndarray) -> str:
    """Describe which equivalent joints differ between wrapped and canonical home.

    Returns a short string like ``"J7:-360→0°"`` or ``"same band"``.
    """
    delta_deg = np.rad2deg(np.abs(wrapped - canonical))
    joint_names = {0: "J1", 2: "J3", 4: "J5", 6: "J7"}
    parts: list[str] = []
    for joint_index, joint_name in joint_names.items():
        if delta_deg[joint_index] > 1.0:
            parts.append(
                f"{joint_name}:{np.rad2deg(wrapped[joint_index]):.0f}→"
                f"{np.rad2deg(canonical[joint_index]):.0f}°"
            )
    return ", ".join(parts) if parts else "same band"


def _latch_operator_estop(
    shared: RuntimeChannels, callback: Callable[[], bool] | None
) -> bool:
    if callback is None:
        return False
    try:
        requested = bool(callback())
    except Exception:
        logger.error("arm home e-stop callback failed", exc_info=True)
        requested = True
    if requested:
        shared.estop_request.value = True
    return requested


def _arm_heartbeat_issue(
    heartbeat_s: float, checked_s: float, max_age_s: float
) -> str | None:
    """Return why an arm heartbeat is unusable, or ``None`` when fresh."""
    if not np.isfinite(heartbeat_s) or heartbeat_s <= 0.0:
        return "arm heartbeat is missing or invalid"
    if not np.isfinite(checked_s):
        return "arm heartbeat check time is invalid"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        return "arm heartbeat maximum age is invalid"
    age_s = checked_s - heartbeat_s
    if age_s < 0.0:
        return f"arm heartbeat is {-age_s:.3f}s in the future"
    if age_s > max_age_s:
        return f"arm heartbeat stale ({age_s:.1f}s)"
    return None


def _estimate_home_timeout_s(
    waypoints: np.ndarray,
    *,
    max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    target_timeout_s: float = arm.homing.target_timeout_s,
) -> float:
    """Deadline derived from milestone path length and feedback settle overhead."""
    if not np.isfinite(max_speed_rad_s) or max_speed_rad_s <= 0.0:
        raise ValueError("homing max speed must be finite and positive")
    if not np.isfinite(target_timeout_s) or target_timeout_s <= 0.0:
        raise ValueError("homing target timeout must be finite and positive")
    if len(waypoints) < 2:
        return _MIN_HOME_TIMEOUT_S
    segment_motion = np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)
    nominal_s = float(np.sum(segment_motion)) / max_speed_rad_s
    moving_segments = int(np.count_nonzero(segment_motion > 1e-9))
    settle_s = moving_segments * target_timeout_s
    return max(
        _MIN_HOME_TIMEOUT_S, 2.0 * nominal_s + settle_s + _HOME_TIMEOUT_PADDING_S
    )


def _wait_for_prehome_state(
    shared: RuntimeChannels,
    *,
    newer_than_ns: int,
    timeout_s: float,
    max_velocity_rad_s: float,
    heartbeat: bool,
    arm_heartbeat_max_age_s: float,
    estop_requested: Callable[[], bool] | None,
) -> tuple[np.ndarray | None, str]:
    """Wait for fresh stationary feedback after invalidating queued actions."""
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("pre-home settle timeout must be finite and positive")
    if not np.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
        raise ValueError("pre-home velocity threshold must be finite and positive")
    deadline_s = time.monotonic() + timeout_s
    consecutive_stationary = 0
    last_stationary_sequence = 0
    while time.monotonic() < deadline_s:
        loop_now_s = time.monotonic()
        if heartbeat:
            shared.set_heartbeat("policy", loop_now_s)
        if _latch_operator_estop(shared, estop_requested):
            return None, "e-stop requested while waiting for stationary feedback"
        if not shared.is_running.value:
            return None, "shutdown requested while waiting for stationary feedback"
        if shared.error_state.value:
            return None, "sticky error_state set while waiting for stationary feedback"
        if int(shared.safety_state.value) == int(SafetyState.FAULT):
            return (
                None,
                "safety state became FAULT while waiting for stationary feedback",
            )
        arm_heartbeat_s = shared.get_heartbeat("arm")
        heartbeat_checked_s = time.monotonic()
        heartbeat_issue = _arm_heartbeat_issue(
            arm_heartbeat_s,
            heartbeat_checked_s,
            arm_heartbeat_max_age_s,
        )
        if heartbeat_issue is not None:
            return None, heartbeat_issue

        state_result = shared.arm_state_ring.read_latest()
        if state_result is not None:
            state, _publish_ns, sequence = state_result
            state_checked_ns = time.monotonic_ns()
            source_ns = int(state["source_monotonic_ns"][0])
            qpos = np.asarray(state["qpos"][0], dtype=np.float64)
            qvel = np.asarray(state["qvel"][0], dtype=np.float64)
            healthy = (
                newer_than_ns <= source_ns <= state_checked_ns
                and bool(state["state_valid"][0])
                and bool(state["connected"][0])
                and int(state["error_code"][0]) == 0
                and qpos.shape == ARM_JOINT_SHAPE
                and qvel.shape == ARM_JOINT_SHAPE
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
            )
            if healthy and float(np.max(np.abs(qvel))) <= max_velocity_rad_s:
                if int(sequence) > last_stationary_sequence:
                    last_stationary_sequence = int(sequence)
                    consecutive_stationary += 1
                    if consecutive_stationary >= 2:
                        return qpos.copy(), ""
            else:
                consecutive_stationary = 0
                last_stationary_sequence = 0
        time.sleep(_HOME_RESULT_POLL_S)
    return None, "stationary arm feedback timed out"


def _format_home_candidate_rejection(candidate: HomePathCandidate) -> str:
    """Format one path-candidate diagnostic without dumping large arrays."""
    name = candidate.name
    reason = candidate.reason or "unknown"
    if reason in ("self_collision", "environment_collision", "collision"):
        pair_names = [
            f"{pair.link1}<->{pair.link2}" for pair in candidate.collision_pairs[:2]
        ]
        pair_text = ",".join(pair_names) if pair_names else "pair unavailable"
        sample_index = (
            candidate.collision_waypoint_index
            if candidate.collision_waypoint_index is not None
            else "?"
        )
        return f"{name}: collision sample={sample_index} ({pair_text})"
    if reason == "table_clearance":
        clearance_mm = 1000.0 * float(
            candidate.clearance_m if candidate.clearance_m is not None else float("nan")
        )
        sample_index = (
            candidate.table_waypoint_index
            if candidate.table_waypoint_index is not None
            else "?"
        )
        return f"{name}: table_clearance sample={sample_index} margin={clearance_mm:+.1f}mm"
    if reason == "workspace":
        segment_index = (
            candidate.workspace_segment_index
            if candidate.workspace_segment_index is not None
            else "?"
        )
        return f"{name}: workspace segment={segment_index}"
    detail = candidate.detail.strip()
    return f"{name}: {reason}" + (f" ({detail})" if detail else "")


def _emit_progress(
    progress: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)


def _home_failure(
    status: ArmHomeStatus,
    detail: str,
    *,
    progress: Callable[[str], None] | None,
    operator_message: str | None = None,
) -> ArmHomeResult:
    _emit_progress(progress, operator_message or f"arm: {detail}")
    return ArmHomeResult(status, detail)


def _wait_for_home_completion(
    shared: RuntimeChannels,
    home_qpos: np.ndarray,
    *,
    newer_than_ns: int,
    timeout_s: float,
    tol_rad: float,
    settled_velocity_rad_s: float,
    heartbeat: bool,
    estop_requested: Callable[[], bool] | None,
    arm_heartbeat_max_age_s: float,
    progress: Callable[[str], None] | None,
) -> ArmHomeResult:
    """Block until the arm worker publishes a fresh stationary frame at ``home_qpos``.

    HOME is a blocking worker operation.  Completion is observed from the arm
    state ring: a frame published after the request whose ``qpos`` converged to
    ``home_qpos`` and whose ``qvel`` has settled.  Milestone frames published
    while the arm is still moving fail the velocity check, so only the settled
    end state (or an already-at-home arm) satisfies it.
    """
    if not np.isfinite(tol_rad) or tol_rad <= 0.0:
        raise ValueError("arm home result tolerance must be finite and positive")
    deadline = time.monotonic() + timeout_s
    abort_reason: str | None = None
    while time.monotonic() < deadline:
        now_s = time.monotonic()
        if heartbeat:
            shared.set_heartbeat("policy", now_s)
        if _latch_operator_estop(shared, estop_requested):
            abort_reason = "e-stop requested by operator"
            break
        if not shared.is_running.value:
            abort_reason = "shutdown requested"
            break
        if shared.error_state.value:
            abort_reason = "sticky error_state set"
            break
        if int(shared.safety_state.value) == int(SafetyState.FAULT):
            abort_reason = "safety state is FAULT"
            break
        if (
            _arm_heartbeat_issue(
                shared.get_heartbeat("arm"), now_s, arm_heartbeat_max_age_s
            )
            is not None
        ):
            abort_reason = "arm heartbeat became stale during homing"
            break
        state_result = shared.arm_state_ring.read_latest()
        if state_result is not None:
            state, _publish_ns, _sequence = state_result
            if int(state["source_monotonic_ns"][0]) > newer_than_ns and bool(
                state["state_valid"][0]
            ):
                qpos = np.asarray(state["qpos"][0], dtype=np.float64)
                qvel = np.asarray(state["qvel"][0], dtype=np.float64)
                if (
                    qpos.shape == ARM_JOINT_SHAPE
                    and qvel.shape == ARM_JOINT_SHAPE
                    and np.all(np.isfinite(qpos))
                    and np.all(np.isfinite(qvel))
                    and float(np.max(np.abs(qpos - home_qpos))) < tol_rad
                    and float(np.max(np.abs(qvel))) <= settled_velocity_rad_s
                ):
                    _emit_progress(progress, "arm: home reached")
                    return ArmHomeResult(ArmHomeStatus.REACHED)
        time.sleep(_HOME_RESULT_POLL_S)
    if abort_reason is not None:
        _emit_progress(progress, f"arm: home wait aborted — {abort_reason}")
        if shared.estop_request.value:
            return ArmHomeResult(ArmHomeStatus.ESTOP_REQUESTED, abort_reason)
        if not shared.is_running.value:
            return ArmHomeResult(ArmHomeStatus.RUNTIME_STOPPED, abort_reason)
        return ArmHomeResult(ArmHomeStatus.FAULTED, abort_reason)
    detail = f"home acknowledgement timed out after {timeout_s:.1f}s"
    _emit_progress(progress, f"arm: {detail}")
    return ArmHomeResult(ArmHomeStatus.COMPLETION_TIMEOUT, detail)


def _resolve_home_waypoints(
    shared: RuntimeChannels,
    current_qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: XArm7MotionPlanner,
    *,
    table_z_surface_m: float,
    estop_requested: Callable[[], bool] | None,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray | None, ArmHomeResult | None]:
    """Select direct or wrapped+alignment milestones without publishing motion."""
    try:
        path_result = compute_joint_home_path(
            current_qpos,
            home_qpos,
            planner,
            table_z_surface_m=table_z_surface_m,
            use_canonical_target=True,
        )
    except Exception as exc:
        logger.warning("execute_arm_home: planning failed", exc_info=True)
        return None, _home_failure(
            ArmHomeStatus.PLANNING_FAILED,
            f"home path planning failed: {exc}",
            progress=progress,
            operator_message=f"arm: home path planning failed — holding ({exc})",
        )

    if _latch_operator_estop(shared, estop_requested):
        return None, _home_failure(
            ArmHomeStatus.ESTOP_REQUESTED,
            "e-stop requested during path planning",
            progress=progress,
            operator_message="arm: homing cancelled during path planning — e-stop requested",
        )
    selected_candidate = path_result.selected_candidate
    waypoints = path_result.waypoints
    if path_result.status is HomePathStatus.UNSAFE:
        canonical_rejections = "; ".join(
            _format_home_candidate_rejection(candidate)
            for candidate in path_result.candidates
            if not candidate.safe
        )
        _emit_progress(
            progress,
            "arm: canonical home path rejected — falling back to wrapped+alignment"
            f" ({canonical_rejections or 'no candidate diagnostics'})",
        )
        try:
            fallback_result = compute_joint_home_path(
                current_qpos,
                home_qpos,
                planner,
                table_z_surface_m=table_z_surface_m,
                use_canonical_target=False,
            )
        except Exception as exc:
            logger.warning("execute_arm_home: fallback planning failed", exc_info=True)
            return None, _home_failure(
                ArmHomeStatus.PLANNING_FAILED,
                f"fallback home path planning failed: {exc}",
                progress=progress,
                operator_message=f"arm: fallback home path planning failed — holding ({exc})",
            )

        if _latch_operator_estop(shared, estop_requested):
            return None, _home_failure(
                ArmHomeStatus.ESTOP_REQUESTED,
                "e-stop requested during fallback path planning",
                progress=progress,
                operator_message=(
                    "arm: homing cancelled during fallback planning — e-stop requested"
                ),
            )
        if fallback_result.status is HomePathStatus.UNSAFE:
            candidate_text = "; ".join(
                _format_home_candidate_rejection(candidate)
                for candidate in fallback_result.candidates
                if not candidate.safe
            )
            qpos_text = np.array2string(
                np.rad2deg(current_qpos), precision=1, separator=","
            )
            return None, _home_failure(
                ArmHomeStatus.NO_SAFE_PATH,
                candidate_text or "no validated home-path candidate",
                progress=progress,
                operator_message=(
                    "arm: no validated home-path candidate — holding\n"
                    f"     current_qpos_deg={qpos_text}\n"
                    f"     rejected={candidate_text or 'no candidate diagnostics'}"
                ),
            )

        waypoints = fallback_result.waypoints
        selected_candidate = fallback_result.selected_candidate
        wrapped_home = (
            waypoints[-1].copy()
            if len(waypoints) > 0
            else planner.ik_mgr.nearest_equivalent_qpos(home_qpos, current_qpos)
        )
        try:
            alignment_result = compute_band_alignment_path(
                wrapped_home,
                home_qpos,
                planner,
                table_z_surface_m=table_z_surface_m,
            )
        except Exception as exc:
            logger.warning(
                "execute_arm_home: band-alignment planning failed", exc_info=True
            )
            return None, _home_failure(
                ArmHomeStatus.PLANNING_FAILED,
                f"band-alignment planning failed: {exc}",
                progress=progress,
                operator_message=f"arm: band-alignment planning failed — holding ({exc})",
            )

        if _latch_operator_estop(shared, estop_requested):
            return None, _home_failure(
                ArmHomeStatus.ESTOP_REQUESTED,
                "e-stop requested during band alignment",
                progress=progress,
                operator_message=(
                    "arm: homing cancelled during band alignment — e-stop requested"
                ),
            )
        if alignment_result.status is HomePathStatus.UNSAFE:
            band_detail = _describe_band_diff(wrapped_home, home_qpos)
            rejection = "; ".join(
                _format_home_candidate_rejection(candidate)
                for candidate in alignment_result.candidates
                if not candidate.safe
            )
            detail = f"band alignment is unsafe: {band_detail}"
            if rejection:
                detail += f" ({rejection})"
            return None, _home_failure(
                ArmHomeStatus.NO_SAFE_PATH,
                detail,
                progress=progress,
                operator_message=f"arm: band-alignment UNSAFE ({band_detail}) — holding",
            )
        if alignment_result.status is HomePathStatus.SAFE:
            alignment_path = alignment_result.waypoints
            tail = alignment_path[1:] if len(waypoints) > 0 else alignment_path
            waypoints = np.concatenate([waypoints, tail], axis=0)
            _emit_progress(
                progress,
                "arm: band-alignment appended "
                f"({len(tail)} milestones, {_describe_band_diff(wrapped_home, home_qpos)})",
            )

    _emit_progress(
        progress,
        f"arm: home path selected={selected_candidate or 'none'} milestones={len(waypoints)}",
    )
    return waypoints, None


def execute_arm_home(
    shared: RuntimeChannels,
    home_qpos: np.ndarray,
    *,
    planner: XArm7MotionPlanner | None,
    config: ArmHomeConfig,
    table_z_surface_m: float = 0.0,
    current_qpos: np.ndarray | None = None,
    estop_requested: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ArmHomeResult:
    """Plan, publish, and observe one arm-home request with typed outcomes."""
    home_qpos = np.asarray(home_qpos, dtype=np.float64)
    if home_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(home_qpos)):
        return _home_failure(
            ArmHomeStatus.INVALID_TARGET,
            "home_qpos must be a finite arm joint vector",
            progress=progress,
            operator_message="arm: invalid home qpos — homing cancelled",
        )
    if _latch_operator_estop(shared, estop_requested):
        return _home_failure(
            ArmHomeStatus.ESTOP_REQUESTED,
            "e-stop requested before homing",
            progress=progress,
            operator_message="arm: homing cancelled — e-stop requested",
        )
    if not shared.is_running.value:
        return _home_failure(
            ArmHomeStatus.RUNTIME_STOPPED,
            "shutdown in progress",
            progress=progress,
            operator_message="arm: homing cancelled — shutdown in progress",
        )
    if shared.error_state.value or int(shared.safety_state.value) == int(
        SafetyState.FAULT
    ):
        return _home_failure(
            ArmHomeStatus.FAULTED,
            "system is in FAULT",
            progress=progress,
            operator_message=(
                "arm: homing cancelled — system is in FAULT; restart after inspection"
            ),
        )
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return _home_failure(
            ArmHomeStatus.NOT_ARMED,
            "safety state is not ARMED",
            progress=progress,
            operator_message="arm: homing cancelled — safety state is not ARMED",
        )
    if current_qpos is not None:
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        if current_qpos.shape != ARM_JOINT_SHAPE or not np.all(
            np.isfinite(current_qpos)
        ):
            return _home_failure(
                ArmHomeStatus.INVALID_CURRENT_STATE,
                "current qpos hint is invalid",
                progress=progress,
                operator_message="arm: invalid current qpos hint — homing cancelled",
            )
    if planner is None:
        return _home_failure(
            ArmHomeStatus.PLANNER_UNAVAILABLE,
            "collision planner is unavailable",
            progress=progress,
            operator_message="arm: no collision planner — homing cancelled",
        )

    if not revoke_motion(shared, SafetyState.ARMED):
        return _home_failure(
            ArmHomeStatus.NOT_ARMED,
            "failed to establish the home command boundary",
            progress=progress,
            operator_message="arm: homing cancelled — command boundary unavailable",
        )
    home_generation = int(shared.run_generation.value)
    generation_started_ns = time.monotonic_ns()
    fresh_qpos, prehome_issue = _wait_for_prehome_state(
        shared,
        newer_than_ns=generation_started_ns,
        timeout_s=max(config.prehome_timeout_s, config.state_max_age_s),
        max_velocity_rad_s=config.stationary_velocity_rad_s,
        heartbeat=config.publish_policy_heartbeat,
        arm_heartbeat_max_age_s=config.arm_heartbeat_max_age_s,
        estop_requested=estop_requested,
    )
    if fresh_qpos is None:
        if shared.estop_request.value:
            status = ArmHomeStatus.ESTOP_REQUESTED
        elif not shared.is_running.value:
            status = ArmHomeStatus.RUNTIME_STOPPED
        elif shared.error_state.value or int(shared.safety_state.value) == int(
            SafetyState.FAULT
        ):
            status = ArmHomeStatus.FAULTED
        else:
            status = ArmHomeStatus.PREHOME_STATE_UNAVAILABLE
        return _home_failure(
            status,
            prehome_issue,
            progress=progress,
            operator_message=(
                "arm: no fresh stationary state after cancelling pending actions — "
                "homing cancelled"
            ),
        )

    waypoints, planning_failure = _resolve_home_waypoints(
        shared,
        fresh_qpos,
        home_qpos,
        planner,
        table_z_surface_m=table_z_surface_m,
        estop_requested=estop_requested,
        progress=progress,
    )
    if planning_failure is not None:
        return planning_failure
    assert waypoints is not None

    if _latch_operator_estop(shared, estop_requested):
        return _home_failure(
            ArmHomeStatus.ESTOP_REQUESTED,
            "e-stop requested before queue publication",
            progress=progress,
            operator_message=(
                "arm: homing cancelled before queue publication — e-stop requested"
            ),
        )
    if int(shared.run_generation.value) != home_generation:
        return _home_failure(
            ArmHomeStatus.GENERATION_CHANGED,
            "run generation changed during planning",
            progress=progress,
            operator_message=(
                "arm: homing cancelled — run generation changed during planning"
            ),
        )

    queued_monotonic_ns = time.monotonic_ns()
    try:
        shared.arm_home_q.put(
            (waypoints, home_qpos.copy(), home_generation),
            timeout=config.request_queue_timeout_s,
        )
    except Full:
        return _home_failure(
            ArmHomeStatus.QUEUE_FULL,
            "home queue is full",
            progress=progress,
            operator_message="arm: home queue is full — homing request was not queued",
        )
    except Exception as exc:
        logger.warning("execute_arm_home: failed to queue HOME request", exc_info=True)
        return _home_failure(
            ArmHomeStatus.QUEUE_ERROR,
            f"failed to queue homing request: {exc}",
            progress=progress,
            operator_message="arm: failed to queue homing request",
        )

    return _wait_for_home_completion(
        shared,
        home_qpos,
        newer_than_ns=queued_monotonic_ns,
        timeout_s=_estimate_home_timeout_s(
            waypoints,
            max_speed_rad_s=config.max_speed_rad_s,
            target_timeout_s=config.target_timeout_s,
        )
        + _RESULT_TIMEOUT_PADDING_S,
        tol_rad=config.result_tolerance_rad,
        settled_velocity_rad_s=config.stationary_velocity_rad_s,
        heartbeat=config.publish_policy_heartbeat,
        estop_requested=estop_requested,
        arm_heartbeat_max_age_s=config.arm_heartbeat_max_age_s,
        progress=progress,
    )


def send_arm_home(
    shared: RuntimeChannels,
    home_qpos: np.ndarray,
    *,
    planner: XArm7MotionPlanner | None = None,
    table_z_surface_m: float = 0.0,
    current_qpos: np.ndarray | None = None,
    queue_timeout: float = arm.homing.request_queue_timeout_s,
    converge_timeout_s: float = arm.homing.convergence_timeout_s,
    state_max_age_s: float = arm.homing.state_max_age_s,
    heartbeat: bool = True,
    estop_requested: Callable[[], bool] | None = None,
    homing_max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    homing_target_timeout_s: float = arm.homing.target_timeout_s,
    arm_heartbeat_max_age_s: float = _ARM_HEARTBEAT_MAX_AGE_S,
    preplan_velocity_rad_s: float = arm.homing.velocity_convergence_rad_s,
    result_tolerance_rad: float = arm.homing.convergence_rad,
    verbose: bool = True,
) -> bool:
    """Compatibility wrapper returning whether the arm reached home."""
    config = ArmHomeConfig(
        request_queue_timeout_s=queue_timeout,
        prehome_timeout_s=converge_timeout_s,
        state_max_age_s=state_max_age_s,
        max_speed_rad_s=homing_max_speed_rad_s,
        target_timeout_s=homing_target_timeout_s,
        arm_heartbeat_max_age_s=arm_heartbeat_max_age_s,
        stationary_velocity_rad_s=preplan_velocity_rad_s,
        result_tolerance_rad=result_tolerance_rad,
        publish_policy_heartbeat=heartbeat,
    )
    progress = (lambda message: print(f"  {message}", flush=True)) if verbose else None
    return execute_arm_home(
        shared,
        home_qpos,
        planner=planner,
        config=config,
        table_z_surface_m=table_z_surface_m,
        current_qpos=current_qpos,
        estop_requested=estop_requested,
        progress=progress,
    ).succeeded

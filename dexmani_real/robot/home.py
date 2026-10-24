"""Planned return-home with full path/environment collision checks."""

import logging
import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning.home_planner import HomePlanner
from dexmani_real.planning.paths import (
    compute_band_alignment_path,
    compute_joint_home_path,
)
from dexmani_real.utils.timing import sample_is_fresh
from dexmani_real.robot.commands import RobotCommand, DispatchError, DispatchInterrupted
from dexmani_real.robot.xhand.limits import validate_hand_command_bounds
from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)
HOME_CONSECUTIVE_SAMPLES = 3


@dataclass(frozen=True)
class HomeResult:
    ok: bool
    reason: str = ""
    interrupted: bool = False


def home_path(current, target, planner, config):
    options = dict(
        table_z_surface_m=config.table_z_surface_m,
        hand_safety_margin_m=config.hand_safety_margin_m,
    )
    try:
        return compute_joint_home_path(
            current, target, planner, use_canonical_target=True, **options
        )
    except ValueError:
        # Retry only when the equivalent representation gives different candidates.
        if np.array_equal(planner.canonicalize_qpos(target, current), target):
            raise
    wrapped = compute_joint_home_path(current, target, planner, **options)
    start = wrapped[-1] if len(wrapped) else planner.canonicalize_qpos(target, current)
    alignment = compute_band_alignment_path(start, target, planner, **options)
    if len(alignment):
        tail = alignment[1:] if len(wrapped) else alignment
        return np.concatenate((wrapped, tail))
    return wrapped


def execute_arm_home(
    runtime_state,
    home_qpos,
    *,
    robot,
    planner,
    config,
    estop_requested=None,
    cancel_requested=None,
    progress=None,
    hand_state_max_age_s=None,
):
    target = np.asarray(home_qpos, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        return HomeResult(False, "home target must be finite (7,)")
    if runtime_state.safety_state != SafetyState.ARMED:
        return HomeResult(False, "home requires ARMED")
    runtime_state.revoke_motion()
    epoch = runtime_state.run_id
    boundary_ns = time.monotonic_ns()

    def aborted():
        if estop_requested is not None and estop_requested():
            runtime_state.request_emergency_stop()
        return (
            cancel_requested is not None and cancel_requested()
        ) or not runtime_state.command_may_cross_sdk(
            run_id=epoch,
            required_safety_state=SafetyState.ARMED,
        )

    deadline = time.monotonic() + config.homing.convergence_timeout_s
    hand_state = None
    feedback_issue = "fresh stationary arm state unavailable"
    while time.monotonic() < deadline:
        if aborted():
            return HomeResult(False, "home interrupted", interrupted=True)
        local = robot.read_state()
        state = local.arm
        feedback_issue = "fresh stationary arm state unavailable"
        if (
            state is not None
            and int(state["timestamp_ns"][0]) > boundary_ns
            and sample_is_fresh(state["timestamp_ns"][0], config.homing.state_max_age_s)
            and np.max(np.abs(state["qvel"][0])) <= config.homing.velocity_convergence_rad_s
        ):
            if hand_state_max_age_s is None:
                break
            hand_state = local.hand
            if (
                hand_state is not None
                and int(hand_state["timestamp_ns"][0]) > boundary_ns
                and sample_is_fresh(hand_state["timestamp_ns"][0], hand_state_max_age_s)
                and np.isfinite(hand_state["qpos"][0]).all()
            ):
                break
            feedback_issue = "fresh post-home hand state unavailable"
        time.sleep(0.01)
    else:
        return HomeResult(False, feedback_issue)
    try:
        if hand_state is not None:
            planner.set_hand_qpos(hand_state["qpos"][0])
        waypoints = home_path(state["qpos"][0], target, planner, config)
    except Exception as exc:
        return HomeResult(False, f"home planning failed: {exc}")
    if aborted():
        return HomeResult(False, "home interrupted during planning", interrupted=True)
    if progress:
        progress(f"arm home: {len(waypoints)} planned milestones")
    travel = (
        float(np.max(np.abs(np.diff(waypoints, axis=0)), axis=1).sum()) if len(waypoints) > 1 else 0
    )
    deadline = time.monotonic() + max(
        10.0,
        2 * travel / config.homing.max_speed_rad_per_s
        + len(waypoints) * config.homing.target_timeout_s
        + 7,
    )
    failure = None
    try:
        ok = robot.home_arm(
            waypoints, target, epoch, lambda: aborted() or time.monotonic() >= deadline
        )
        interrupted = not ok and time.monotonic() < deadline
        return HomeResult(
            ok,
            "" if ok else "home interrupted" if interrupted else "arm home timed out",
            interrupted=interrupted,
        )
    except BaseException as exc:
        failure = exc
        from dexmani_real.robot.xarm7.protocol import OperationCancelled

        if isinstance(exc, KeyboardInterrupt):
            runtime_state.request_emergency_stop()
            raise
        if not isinstance(exc, Exception):
            raise
        if not isinstance(exc, OperationCancelled):
            runtime_state.mark_error()
            runtime_state.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        interrupted = time.monotonic() < deadline
        failure = None  # OperationCancelled is handled; a cleanup failure must still propagate.
        return HomeResult(
            False, str(exc) if interrupted else "arm home timed out", interrupted=interrupted
        )
    finally:
        try:
            robot.stop()
        except Exception:
            if failure is None:
                raise

            logger.exception("HOME cleanup also failed")


def build_home_planner(runtime: ExperimentConfig) -> HomePlanner:
    """Build HOME planning with current workspace/table/static geometry beyond online IK endpoint checks."""
    return HomePlanner(
        workspace_bounds=runtime.environment.workspace.as_array(),
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )


def home_robot(runtime_state, runtime, planner, *, robot, abort_requested) -> HomeResult:
    if runtime_state.safety_state != SafetyState.ARMED:
        return HomeResult(False, "home requires ARMED")
    failure = None
    try:
        hand_result = home_hand(
            runtime_state, runtime, robot=robot, abort_requested=abort_requested
        )
        if not hand_result.ok:
            return hand_result
        # Hand-disabled mode assumes the hand is absent or secured at home.
        if not runtime.hand.enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
        result = execute_arm_home(
            runtime_state,
            runtime.arm.home_qpos,
            robot=robot,
            planner=planner,
            config=runtime.arm,
            cancel_requested=abort_requested,
            estop_requested=lambda: bool(runtime_state.estop_request),
            progress=logger.debug,
            hand_state_max_age_s=(
                runtime.hand.feedback_max_age_s if runtime.hand.enabled else None
            ),
        )
        return result
    except BaseException as exc:
        failure = exc
        if isinstance(exc, KeyboardInterrupt):
            runtime_state.request_emergency_stop()
        raise
    finally:
        try:
            # execute_arm_home owns its completed stop; retry only unresolved motion.
            if robot.stop_required:
                robot.stop()
        except Exception:
            if failure is None:
                raise

            logger.exception("combined HOME cleanup also failed")


def home_hand(runtime_state, runtime, *, robot, abort_requested=None):
    if not runtime.hand.enabled:
        return HomeResult(True)
    cfg = runtime.hand
    target = validate_hand_command_bounds(
        np.deg2rad(cfg.home_qpos_deg),
        np.asarray(cfg.qpos_min_rad),
        np.asarray(cfg.qpos_max_rad),
        np.asarray(cfg.mechanical_qpos_min_rad),
        np.asarray(cfg.mechanical_qpos_max_rad),
    )
    if runtime_state.safety_state != SafetyState.ARMED:
        return HomeResult(False, "hand home requires ARMED")
    runtime_state.revoke_motion()
    epoch = runtime_state.run_id
    if not runtime_state.command_may_cross_sdk(
        run_id=epoch, required_safety_state=SafetyState.ARMED
    ):
        return HomeResult(False, "hand home requires ARMED authority", interrupted=True)
    if abort_requested is not None and abort_requested():
        return HomeResult(False, "home interrupted", interrupted=True)
    deadline = time.monotonic_ns() + int(cfg.home_timeout_s * 1e9)
    try:
        robot.send_hand_home(RobotCommand(epoch, hand_qpos=target), valid_until_ns=deadline)
    except DispatchInterrupted as exc:
        logger.warning("hand HOME cancelled: dispatch=%s", exc.result)
        raise
    except DispatchError as exc:
        if not exc.revoked:
            raise
        logger.info("hand HOME dispatch stopped: %s; dispatch=%s", exc, exc.result)
        runtime_state.revoke_motion(expected_run_id=epoch)
        robot.stop()
        if exc.cause != "authority_revoked":
            raise
        return HomeResult(False, "home interrupted", interrupted=True)

    # Only feedback newer than the submission result can establish arrival.
    last_sample_ns = time.monotonic_ns()
    consecutive = 0
    tolerance_rad = np.deg2rad(cfg.home_tolerance_deg)
    interrupted = False
    while time.monotonic_ns() < deadline:
        if abort_requested is not None and abort_requested():
            interrupted = True
            break
        if not runtime_state.command_may_cross_sdk(
            run_id=epoch, required_safety_state=SafetyState.ARMED
        ):
            interrupted = True
            break
        local = robot.read_state()
        if local.hand is not None:
            sample = local.hand[0]
            stamp = int(sample["timestamp_ns"])
            if stamp > last_sample_ns:
                last_sample_ns = stamp
                qpos = sample["qpos"]
                if (
                    sample_is_fresh(stamp, cfg.feedback_max_age_s)
                    and np.isfinite(qpos).all()
                    and np.max(np.abs(qpos - target)) <= tolerance_rad
                ):
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= HOME_CONSECUTIVE_SAMPLES:
                    if (
                        time.monotonic_ns() < deadline
                        and not (abort_requested is not None and abort_requested())
                        and runtime_state.command_may_cross_sdk(
                            run_id=epoch, required_safety_state=SafetyState.ARMED
                        )
                    ):
                        return HomeResult(True)
                    interrupted = time.monotonic_ns() < deadline
                    break
        time.sleep(0.01)
    runtime_state.revoke_motion(expected_run_id=epoch)
    robot.stop()
    return HomeResult(
        False,
        "home interrupted" if interrupted else "hand home convergence timed out",
        interrupted=interrupted,
    )

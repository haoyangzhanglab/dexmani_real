"""Planned return-home with full path/environment collision checks."""

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.kinematics.ik import make_online_ik_config
from dexmani_real.planning.paths import (
    HomePathStatus,
    compute_band_alignment_path,
    compute_joint_home_path,
)
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.home import HomeResult
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    command_may_cross_sdk,
    revoke_motion,
)


@dataclass(frozen=True)
class ArmHomeConfig:
    prehome_timeout_s: float
    state_max_age_s: float
    max_speed_rad_s: float
    target_timeout_s: float
    stationary_velocity_rad_s: float
    table_z_surface_m: float
    hand_safety_margin_m: float

    @classmethod
    def from_runtime(cls, runtime):
        h = runtime.arm.homing
        return cls(
            h.convergence_timeout_s,
            h.state_max_age_s,
            h.max_speed_rad_per_s,
            h.target_timeout_s,
            h.velocity_convergence_rad_s,
            runtime.arm.table_z_surface_m,
            runtime.arm.hand_safety_margin_m,
        )


def _home_path(current, target, planner, config):
    options = dict(
        table_z_surface_m=config.table_z_surface_m,
        hand_safety_margin_m=config.hand_safety_margin_m,
    )
    path = compute_joint_home_path(current, target, planner, use_canonical_target=True, **options)
    if path.status is not HomePathStatus.UNSAFE:
        return path.waypoints
    wrapped = compute_joint_home_path(
        current, target, planner, use_canonical_target=False, **options
    )
    if wrapped.status is HomePathStatus.UNSAFE:
        raise ValueError(f"no safe home path: {wrapped.candidates}")
    start = (
        wrapped.waypoints[-1]
        if len(wrapped.waypoints)
        else planner.ik_geometry.nearest_equivalent_qpos(target, current)
    )
    alignment = compute_band_alignment_path(start, target, planner, **options)
    if alignment.status is HomePathStatus.UNSAFE:
        raise ValueError(f"no safe equivalent-angle alignment: {alignment.candidates}")
    if alignment.status is HomePathStatus.SAFE:
        tail = alignment.waypoints[1:] if len(wrapped.waypoints) else alignment.waypoints
        return np.concatenate((wrapped.waypoints, tail))
    return wrapped.waypoints


def execute_arm_home(
    shared,
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
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return HomeResult(False, "home requires ARMED")
    revoke_motion(shared)
    epoch = int(shared.run_id.value)
    boundary_ns = time.monotonic_ns()

    def aborted():
        if estop_requested is not None and estop_requested():
            shared.estop_request.value = True
        return (cancel_requested is not None and cancel_requested()) or not command_may_cross_sdk(
            shared, run_id=epoch, required_safety_state=SafetyState.ARMED
        )

    deadline = time.monotonic() + config.prehome_timeout_s
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
            and sample_is_fresh(state["timestamp_ns"][0], config.state_max_age_s)
            and np.max(np.abs(state["qvel"][0])) <= config.stationary_velocity_rad_s
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
        waypoints = _home_path(state["qpos"][0], target, planner, config)
    except Exception as exc:
        return HomeResult(False, f"home planning failed: {exc}")
    if aborted():
        return HomeResult(False, "home interrupted during planning", interrupted=True)
    if progress:
        progress(f"arm home: {len(waypoints)} planned milestones")
    # Bound total HOME duration using path travel and per-milestone settling allowances.
    travel = (
        float(np.max(np.abs(np.diff(waypoints, axis=0)), axis=1).sum()) if len(waypoints) > 1 else 0
    )
    deadline = time.monotonic() + max(
        10.0, 2 * travel / config.max_speed_rad_s + len(waypoints) * config.target_timeout_s + 7
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
        from dexmani_real.robot.drivers.xarm7 import HomeAborted

        if isinstance(exc, KeyboardInterrupt):
            shared.estop_request.value = True
            raise
        if not isinstance(exc, Exception):
            raise
        if not isinstance(exc, HomeAborted):
            shared.error_state.value = True
            revoke_motion(shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        interrupted = time.monotonic() < deadline
        failure = None  # HomeAborted is handled; a cleanup failure must still propagate.
        return HomeResult(
            False, str(exc) if interrupted else "arm home timed out", interrupted=interrupted
        )
    finally:
        try:
            robot.stop()
        except Exception:
            if failure is None:
                raise
            from dexmani_real.utils.log import get_logger

            get_logger(__name__).exception("HOME cleanup also failed")


def build_policy_home_planner(runtime: ExperimentConfig) -> XArm7MotionPlanner:
    """Construct the shared return-home planner for teleop, policy and replay.

    Online Cartesian IK checks robot endpoint self-collision. Return-home
    additionally needs path/workspace/table/static-box checks, so HOME uses
    a planner configured with the current environment.
    """
    policy = runtime.policy
    workspace = policy.workspace.as_array()
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        online_ik_profile=make_online_ik_config(runtime),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )


def home_policy_robot(shared, runtime, planner, *, robot, abort_requested):
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return False
    failure = None
    try:
        hand_result = home_hand(shared, runtime, robot=robot, abort_requested=abort_requested)
        if not hand_result.ok:
            print(f"Hand home failed: {hand_result.reason}", flush=True)
            return False
        # Hand-disabled mode assumes the hand is absent or secured at home.
        if not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
        result = execute_arm_home(
            shared,
            runtime.arm.home_qpos,
            robot=robot,
            planner=planner,
            config=ArmHomeConfig.from_runtime(runtime),
            cancel_requested=abort_requested,
            estop_requested=lambda: bool(shared.estop_request.value),
            progress=print,
            hand_state_max_age_s=(
                runtime.hand.feedback_max_age_s if runtime.policy.hand_enabled else None
            ),
        )
        if not result.ok:
            print(f"Home failed: {result.reason}", flush=True)
        return result.ok
    except BaseException as exc:
        failure = exc
        if isinstance(exc, KeyboardInterrupt):
            shared.estop_request.value = True
        raise
    finally:
        try:
            # execute_arm_home owns its completed stop; retry only unresolved motion.
            if robot.stop_required:
                robot.stop()
        except Exception:
            if failure is None:
                raise
            from dexmani_real.utils.log import get_logger

            get_logger(__name__).exception("combined HOME cleanup also failed")

"""Planned return-home; collision geometry stays here, outside streaming control."""
import time
from queue import Full, Empty
from dataclasses import dataclass
import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import read_arm_state
from dexmani_real.planning import OnlineIKConfig, Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.paths import HomePathStatus, compute_joint_home_path, compute_band_alignment_path
from dexmani_real.robot.model import XARM7_XHAND_COLLISION_URDF_PATH, XARM7_XHAND_SRDF_PATH
from dexmani_real.robot.home import HomeResult, wait_home_result
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk, revoke_motion


@dataclass(frozen=True)
class ArmHomeConfig:
    request_queue_timeout_s: float
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
        return cls(h.request_queue_timeout_s, h.convergence_timeout_s, h.state_max_age_s,
                   h.max_speed_rad_per_s, h.target_timeout_s, h.velocity_convergence_rad_s,
                   runtime.arm.table_z_surface_m, runtime.arm.hand_safety_margin_m)


def _home_path(current, target, planner, config):
    options = dict(table_z_surface_m=config.table_z_surface_m,
                   hand_safety_margin_m=config.hand_safety_margin_m)
    path = compute_joint_home_path(current, target, planner, use_canonical_target=True, **options)
    if path.status is not HomePathStatus.UNSAFE:
        return path.waypoints
    wrapped = compute_joint_home_path(current, target, planner, use_canonical_target=False, **options)
    if wrapped.status is HomePathStatus.UNSAFE:
        raise ValueError(f"no safe home path: {wrapped.candidates}")
    start = wrapped.waypoints[-1] if len(wrapped.waypoints) else planner.ik_mgr.nearest_equivalent_qpos(target, current)
    alignment = compute_band_alignment_path(start, target, planner, **options)
    if alignment.status is HomePathStatus.UNSAFE:
        raise ValueError(f"no safe equivalent-angle alignment: {alignment.candidates}")
    if alignment.status is HomePathStatus.SAFE:
        tail = alignment.waypoints[1:] if len(wrapped.waypoints) else alignment.waypoints
        return np.concatenate((wrapped.waypoints, tail))
    return wrapped.waypoints


def execute_arm_home(shared, home_qpos, *, planner, config, estop_requested=None,
                     cancel_requested=None, progress=None):
    target = np.asarray(home_qpos, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        return HomeResult(False, "home target must be finite (7,)")
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return HomeResult(False, "home requires ARMED")
    revoke_motion(shared)
    epoch = int(shared.run_id.value)
    while True:
        try:
            shared.arm_home_result_q.get_nowait()
        except Empty:
            break
    boundary_ns = time.monotonic_ns()
    def aborted():
        if estop_requested is not None and estop_requested():
            shared.estop_request.value = True
        return ((cancel_requested is not None and cancel_requested()) or
                not command_may_cross_sdk(shared, run_id=epoch, required_safety_state=SafetyState.ARMED))
    deadline = time.monotonic() + config.prehome_timeout_s
    while time.monotonic() < deadline:
        if aborted():
            return HomeResult(False, "home interrupted")
        state = read_arm_state(shared)
        if (state is not None and int(state["timestamp_ns"][0]) > boundary_ns
                and sample_is_fresh(state["timestamp_ns"][0], config.state_max_age_s)
                and np.max(np.abs(state["qvel"][0])) <= config.stationary_velocity_rad_s):
            break
        time.sleep(0.01)
    else:
        return HomeResult(False, "fresh stationary arm state unavailable")
    try:
        waypoints = _home_path(state["qpos"][0], target, planner, config)
    except Exception as exc:
        return HomeResult(False, f"home planning failed: {exc}")
    if aborted():
        return HomeResult(False, "home interrupted during planning")
    if progress:
        progress(f"arm home: {len(waypoints)} planned milestones")
    try:
        shared.arm_home_q.put_nowait((waypoints, target, epoch,
            time.monotonic_ns() + int(config.request_queue_timeout_s * 1e9)))
    except Full:
        return HomeResult(False, "arm home queue full")
    travel = float(np.max(np.abs(np.diff(waypoints, axis=0)), axis=1).sum()) if len(waypoints) > 1 else 0
    timeout = max(10.0, 2 * travel / config.max_speed_rad_s + len(waypoints) * config.target_timeout_s + 7)
    return wait_home_result(shared, shared.arm_home_result_q, epoch, timeout, aborted)


def build_policy_home_planner(runtime: ExperimentConfig) -> XArm7MotionPlanner:
    """Construct the collision-checked home planner (mirrors replay setup).

    The policy runner's planner only carries workspace bounds; a safe
    return-home needs the full collision model (hand-dof + table + static
    boxes), so the Main process builds its own.
    """
    policy = runtime.policy
    workspace = policy.workspace.as_array()
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        teleop_profile=OnlineIKConfig(
            max_pose_error_pos_m=float(policy.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(policy.ik_max_pose_error_rot_rad),
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )


def home_policy_robot(shared, runtime, planner, *, abort_requested):
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return False
    hand_result = home_hand(shared, runtime, abort_requested=abort_requested)
    if not hand_result.ok:
        print(f"Hand home failed: {hand_result.reason}", flush=True)
        return False
    # Home geometry uses the commanded hand posture; SDK success is not arrival.
    planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
    result = execute_arm_home(shared, runtime.arm.home_qpos, planner=planner,
        config=ArmHomeConfig.from_runtime(runtime), cancel_requested=abort_requested,
        estop_requested=lambda: bool(shared.estop_request.value), progress=print)
    if not result.ok:
        print(f"Home failed: {result.reason}", flush=True)
    return result.ok

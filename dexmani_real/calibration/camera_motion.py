"""Interactive arm motion for the camera-calibration experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning import Pose
from dexmani_real.planning.pose_utils import quat_multiply, rot6d_to_quat_wxyz
from dexmani_real.policy.action_protocol import publish_joint_targets
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition
from dexmani_real.shm.shared_storage import SharedStorage, read_arm_state_dict
from dexmani_real.teleop.keyboard import GlobalKeyState, MotionActivityLatch, eef_delta_from_keys, validate_arm_feedback
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.calibration.camera_session import CameraCalibrationSession

logger = get_logger(__name__)

_AXIS_NAMES = ("x", "y", "z")


def directional_workspace_step(
    target_pos_m: object,
    delta_pos_m: object,
    workspace_bounds_m: object,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply a soft wall while allowing an out-of-bounds target to move inward."""
    target = np.asarray(target_pos_m, dtype=np.float64)
    delta = np.asarray(delta_pos_m, dtype=np.float64)
    bounds = np.asarray(workspace_bounds_m, dtype=np.float64)
    if target.shape != (3,) or delta.shape != (3,) or bounds.shape != (3, 2):
        raise ValueError("workspace step expects target/delta (3,) and bounds (3, 2)")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(bounds)):
        raise ValueError("workspace step inputs must be finite")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("workspace lower bounds must not exceed upper bounds")

    result = target.copy()
    proposed = target + delta
    rejected: list[int] = []
    for axis in range(3):
        if delta[axis] == 0.0:
            continue
        lower, upper = bounds[axis]
        if lower <= proposed[axis] <= upper:
            result[axis] = proposed[axis]
        elif (target[axis] < lower and delta[axis] > 0.0) or (target[axis] > upper and delta[axis] < 0.0):
            result[axis] = float(np.clip(proposed[axis], lower, upper))
        else:
            rejected.append(axis)
    return result, tuple(rejected)


@dataclass
class MotionState:
    arm_qpos: np.ndarray
    eef_pos_base_m: np.ndarray
    eef_rot6d_base: np.ndarray
    previous_qpos_command: np.ndarray
    target_pos_world_m: np.ndarray
    target_quat_world_wxyz: np.ndarray
    motion_latch: MotionActivityLatch = field(default_factory=MotionActivityLatch)
    state_error_count: int = 0
    loop_count: int = 0
    previous_home_pressed: bool = False
    wall_warned: list[bool] = field(default_factory=lambda: [False, False, False])
    wall_warning_time_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class CameraMotionController:
    """Map keyboard deltas to geometry-gated arm commands."""

    def __init__(self, session: CameraCalibrationSession, initial_state: dict[str, Any]) -> None:
        self.session = session
        self.runtime = session.runtime
        self.config = session.config
        self.planner = session.planner
        shared = session.shared
        keys = session.keys
        if shared is None or keys is None or session.arm_config is None:
            raise RuntimeError("camera calibration session is not ready for motion")
        self.shared: SharedStorage = shared
        self.keys: GlobalKeyState = keys
        self.arm_config = session.arm_config
        self.recoverable_errors = frozenset(int(code) for code in self.runtime.arm.recoverable_errors)
        self.collision_errors = frozenset(int(code) for code in self.runtime.arm.collision_fault_errors)
        self.state = self._make_motion_state(initial_state)

    def _make_motion_state(self, state: dict[str, Any]) -> MotionState:
        arm_qpos = np.asarray(state["qpos"], dtype=np.float64).copy()
        eef_pos = np.asarray(state["eef_pos"], dtype=np.float64).copy()
        eef_rot6d = np.asarray(state["eef_rot6d"], dtype=np.float64).copy()
        eef_world = self.planner.base_to_world_pose(Pose(p=eef_pos, q=rot6d_to_quat_wxyz(eef_rot6d)))
        print(f"\n  当前 EEF (world): pos={np.round(eef_world.p, 4)}m  q={np.round(eef_world.q, 4)}")
        return MotionState(
            arm_qpos=arm_qpos,
            eef_pos_base_m=eef_pos,
            eef_rot6d_base=eef_rot6d,
            previous_qpos_command=arm_qpos.copy(),
            target_pos_world_m=eef_world.p.copy(),
            target_quat_world_wxyz=eef_world.q.copy(),
        )

    def _sync_feedback(self, state: dict[str, Any]) -> None:
        self.state.arm_qpos = np.asarray(state["qpos"], dtype=np.float64).copy()
        self.state.eef_pos_base_m = np.asarray(state["eef_pos"], dtype=np.float64).copy()
        self.state.eef_rot6d_base = np.asarray(state["eef_rot6d"], dtype=np.float64).copy()
        self.state.previous_qpos_command = self.state.arm_qpos.copy()
        eef_world = self.planner.base_to_world_pose(
            Pose(
                p=self.state.eef_pos_base_m,
                q=rot6d_to_quat_wxyz(self.state.eef_rot6d_base),
            )
        )
        self.state.target_pos_world_m = eef_world.p.copy()
        self.state.target_quat_world_wxyz = eef_world.q.copy()

    def handle_home(self) -> bool:
        pressed = self.keys.is_pressed("r")
        requested = pressed and not self.state.previous_home_pressed
        self.state.previous_home_pressed = pressed
        if not requested:
            return False
        print("\n  R: return_home")
        if int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
            require_transition(self.shared, SafetyState.ARMED)
        home_ok = send_arm_home(
            self.shared,
            np.asarray(self.arm_config.home_qpos, dtype=np.float64),
            planner=self.planner,
            table_z_surface_m=float(self.runtime.arm.table_z_surface_m),
            current_qpos=self.state.arm_qpos,
            queue_timeout=float(self.runtime.arm.homing.request_queue_timeout_s),
            converge_timeout_s=float(self.runtime.arm.homing.convergence_timeout_s),
            state_max_age_s=float(self.runtime.arm.homing.state_max_age_s),
            heartbeat=False,
            estop_requested=lambda: self.keys.is_pressed("esc"),
            homing_max_speed_rad_s=float(np.deg2rad(self.runtime.arm.homing.max_speed_deg_s)),
            homing_target_timeout_s=float(self.runtime.arm.homing.target_timeout_s),
            arm_heartbeat_max_age_s=float(self.runtime.safety.heartbeat_timeouts["arm"]),
            preplan_velocity_rad_s=float(self.runtime.arm.homing.velocity_convergence_rad_s),
            verbose=True,
        )
        if self.shared.estop_request.value:
            self.session.latch_estop("e-stop during arm homing")
            return True
        if not home_ok:
            if self.shared.error_state.value or int(self.shared.safety_state.value) == int(SafetyState.FAULT):
                self.session.latch_fault("arm homing failed with a shared fault")
            else:
                print("  ⚠ return_home 未执行，机械臂保持当前位置")
            return True

        self.state.motion_latch.reset()
        state, issue = self.session.read_capture_state()
        if state is None:
            print(f"  ⚠ 归位后状态无效: {issue}")
            return True
        try:
            self._sync_feedback(state)
        except (RuntimeError, ValueError):
            logger.warning("Failed to synchronize post-home EEF state", exc_info=True)
            return True
        print("  Arm 归位完成，状态已同步")
        return True

    def _read_control_feedback(self) -> dict[str, Any] | None:
        state = read_arm_state_dict(self.shared)
        issue: str | None = "arm state unavailable"
        if state is not None:
            issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(self.runtime.policy.arm_state_stale_threshold_s),
                qpos=state["qpos"],
                qvel=state["qvel"],
                eef_pos=state["eef_pos"],
                eef_rot6d=state["eef_rot6d"],
            )
        if issue is not None:
            self.state.state_error_count += 1
            if self.state.state_error_count >= int(self.runtime.policy.max_consecutive_errors):
                logger.error("Persistent invalid arm feedback: %s", issue)
                self.session.latch_estop("persistent invalid arm feedback")
            return None
        assert state is not None
        error_code = int(state["error_code"])
        if error_code in self.recoverable_errors:
            return None
        if error_code != 0:
            category = "collision" if error_code in self.collision_errors else "controller"
            logger.error("Arm %s error C%d", category, error_code)
            self.session.latch_estop(f"arm {category} error C{error_code}")
            return None
        return state

    def _world_feedback(self, state: dict[str, Any]) -> Pose | None:
        self.state.arm_qpos = np.asarray(state["qpos"], dtype=np.float64)
        self.state.eef_pos_base_m = np.asarray(state["eef_pos"], dtype=np.float64)
        self.state.eef_rot6d_base = np.asarray(state["eef_rot6d"], dtype=np.float64)
        try:
            pose = self.planner.base_to_world_pose(
                Pose(
                    p=self.state.eef_pos_base_m,
                    q=rot6d_to_quat_wxyz(self.state.eef_rot6d_base),
                )
            )
        except (RuntimeError, ValueError):
            self.state.state_error_count += 1
            logger.warning("EEF pose conversion failed", exc_info=True)
            return None
        self.state.state_error_count = 0
        return pose

    def _warn_workspace(self, rejected_axes: tuple[int, ...]) -> None:
        now_s = time.perf_counter()
        for axis in rejected_axes:
            if (
                not self.state.wall_warned[axis]
                or now_s - self.state.wall_warning_time_s[axis] > self.config.wall_warning_interval_s
            ):
                lower, upper = self.session.workspace_bounds_m[axis]
                print(f"  ⚠ {_AXIS_NAMES[axis]} 边界 [{lower:.2f}, {upper:.2f}]")
                self.state.wall_warned[axis] = True
                self.state.wall_warning_time_s[axis] = now_s

    def _apply_motion_delta(self, eef_world: Pose, delta_pos: np.ndarray, delta_rpy: np.ndarray) -> None:
        lead_m = float(np.linalg.norm(self.state.target_pos_world_m - eef_world.p))
        if lead_m > self.config.target_lead_max_m:
            direction = self.state.target_pos_world_m - eef_world.p
            self.state.target_pos_world_m = eef_world.p + direction * (self.config.target_lead_max_m / lead_m)
        self.state.target_pos_world_m, rejected = directional_workspace_step(
            self.state.target_pos_world_m,
            delta_pos,
            self.session.workspace_bounds_m,
        )
        self._warn_workspace(rejected)
        if np.any(delta_rpy != 0.0):
            delta_quat = Rotation.from_euler("xyz", delta_rpy).as_quat(scalar_first=True)
            self.state.target_quat_world_wxyz = quat_multiply(
                delta_quat,
                self.state.target_quat_world_wxyz,
            )

    def _publish_target(self, eef_world: Pose) -> None:
        target = Pose(p=self.state.target_pos_world_m, q=self.state.target_quat_world_wxyz)
        result = self.planner.solve_teleop_ik(
            target,
            self.state.arm_qpos,
            self.state.previous_qpos_command,
        )
        if not result.success or result.qpos is None:
            self.state.target_pos_world_m = eef_world.p.copy()
            self.state.target_quat_world_wxyz = eef_world.q.copy()
            return
        if not np.all(np.isfinite(result.qpos)) or int(self.shared.safety_state.value) == int(SafetyState.FAULT):
            return
        candidate = publish_joint_targets(
            self.shared,
            result.qpos,
            prepare_timeout_s=float(self.runtime.policy.action_prepare_timeout_s),
            dt_s=1.0 / self.session.control_hz,
            safety_gate=self.session.action_safety_gate,
        )
        if candidate is None:
            self.session.latch_fault("arm command was rejected or could not be committed")
            return
        assert candidate.arm_qpos is not None
        self.state.previous_qpos_command = np.asarray(candidate.arm_qpos, dtype=np.float64)

    def step(self, sample_count: int) -> None:
        self.state.loop_count += 1
        feedback = self._read_control_feedback()
        if feedback is None:
            return
        eef_world = self._world_feedback(feedback)
        if eef_world is None:
            return
        delta_pos, delta_rpy = eef_delta_from_keys(
            self.keys,
            self.config.delta_pos_m,
            self.config.delta_rpy_rad,
        )
        if self.state.loop_count % self.config.status_interval_frames == 0:
            hint = "← 按 SPACE 采集" if sample_count < self.config.min_samples else "← 按 ENTER 标定"
            print(
                f"[{self.state.loop_count:5d}] eef_w={np.round(eef_world.p, 3)}m  " f"samples={sample_count}  {hint}",
                flush=True,
            )

        moving = bool(np.any(delta_pos != 0.0) or np.any(delta_rpy != 0.0))
        if not moving:
            released = self.state.motion_latch.update(False)
            if released and int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(self.shared, SafetyState.ARMED)
            if released:
                hold_pose = self.planner.kin.compute_eef_pose_world(self.state.previous_qpos_command)
                self.state.target_pos_world_m = hold_pose.p.copy()
                self.state.target_quat_world_wxyz = hold_pose.q.copy()
            return

        self.state.motion_latch.update(True)
        if int(self.shared.safety_state.value) == int(SafetyState.ARMED):
            require_transition(self.shared, SafetyState.RUNNING)
        self._apply_motion_delta(eef_world, delta_pos, delta_rpy)
        self._publish_target(eef_world)


__all__ = ["CameraMotionController", "MotionState", "directional_workspace_step"]

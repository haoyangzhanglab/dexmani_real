"""RobotInterface — arm + hand unified interface.

Controllers and deployment modules operate hardware exclusively through
RobotInterface, never calling XArm7/XHand directly.
"""

from __future__ import annotations

import signal as _signal
import time
import traceback
import warnings
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.planning.kinematics import XArm7Kinematics
from dexmani_real.planning.types import Pose
from dexmani_real.planning.pose_utils import compose_pose, compute_pose_error, quat_wxyz_to_rot6d
from dexmani_real.planning import WorkspaceSafety
from dexmani_real.robot.hand_kinematics import HandKinematics
from dexmani_real.robot.types import RobotAction, RobotInterfaceConfig, RobotState
from dexmani_real.teleop.control.safety import _ARM_TORQUE_LIMIT_NM
from dexmani_real.robot.xarm7 import XArm7
from dexmani_real.robot.xhand import XHand
from dexmani_real.utils.array_utils import nan_array

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Magic numbers (Phase 4.2)
# ---------------------------------------------------------------------------

_HAND_RESET_CONVERGE_TIMEOUT_S = 3.0
_HAND_RESET_CONVERGE_THRESHOLD_RAD = np.deg2rad(5.0)
_HAND_RESET_POLL_INTERVAL_S = 0.05

_PHASE1_CONVERGE_THRESHOLD_RAD = np.deg2rad(3.0)
_PHASE1_POLL_INTERVAL_MULT = 2  # × dt
_PHASE1_MAX_WAIT_MULT = 5  # × theoretical travel time

_PHASE2_MIN_DELTA_RAD = np.deg2rad(0.5)

_HOME_JOINT_THRESHOLD_RAD = np.deg2rad(3.0)  # matches _PHASE1_CONVERGE_THRESHOLD_RAD; arm.reset() servo tolerance

# Safety lift geometry (Phase 4.2) — used by _lift_eef_z_safe()
_HAND_EXTENSION_Z_M = 0.076         # max fingertip-to-EEF vertical offset when hand is extended
_HAND_SAFETY_MARGIN_Z_M = 0.03     # additional clearance above table+hand_extension
_JOINT_DIP_ALLOWANCE_Z_M = 0.05     # extra margin for joint-space trajectory dip below EEF Z
_Z_LIFT_EXTRA_MARGIN_M = 0.05       # additional lift past min_safe_z toward workspace midpoint
_Z_LIFT_WS_TOP_MARGIN_M = 0.02      # margin below workspace ceiling to avoid singularity


def _dense_interpolate(path: np.ndarray, max_step_rad: float = np.deg2rad(1.0)) -> np.ndarray:
    """Densify a sparse joint path so each step ≤ max_step_rad."""
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step_rad))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


class RobotInterface:
    """Arm + Hand unified interface.

    - Controllers and deployment modules operate hardware exclusively through this class.
    - Degraded operation when hand disconnects (arm still works).
    - send_action() returns dict[str, bool] with per-device status.
    """

    def __init__(
        self,
        config: RobotInterfaceConfig,
        kinematics: XArm7Kinematics,
        *,
        planner: XArm7MotionPlanner | None = None,
    ) -> None:
        self.config = config
        self.kinematics = kinematics
        self.planner = planner
        self.workspace = WorkspaceSafety(
            config.workspace_bounds,
            orientation_bounds=config.workspace_orientation_bounds,
        )

        self.arm = XArm7(config.arm)
        self.hand = XHand(config.hand)

        # Validate home EEF is within workspace (catch config errors early)
        home_pose = self.kinematics.compute_eef_pose_world(self.arm.config.init_qpos)
        if not self.workspace.check(home_pose.p):
            msg = (
                f"init_qpos FK yields EEF {np.round(home_pose.p, 4)} m "
                f"outside workspace bounds {self.workspace.bounds}. "
                f"Fix init_qpos, base_pose_world, or workspace_bounds."
            )
            if np.all(np.isfinite(home_pose.p)):
                raise ValueError(msg)
            else:
                warnings.warn(f"Cannot validate home EEF workspace (NaN FK): {msg}")

        # Set up table collision geometry (plan_screw/plan_qpos will avoid).
        # Only add MPlib point cloud when env_collision_mode == "mplib_pointcloud".
        # Default "geometric_fk" mode uses FingertipDeskSafety (FK fingertip Z)
        # which is zero-cost and does NOT pollute the IK solver.
        if self.planner is not None:
            use_pointcloud = False
            if config.collision is not None:
                use_pointcloud = config.collision.env_collision_mode == "mplib_pointcloud"
            elif config.add_table_collision:
                # Backward compat: old config without CollisionConfig.
                # Default to geometric_fk (zero-cost, no IK pollution) instead of
                # MPlib point cloud which costs ~47% IK success rate (100% → 53%).
                # Auto-create CollisionConfig and inject into planner so FK desk
                # safety is active even for consumers that haven't migrated yet.
                from dexmani_real.planning.collision_config import CollisionConfig

                auto_collision = CollisionConfig(
                    table_z_world=config.table_z_world,
                    env_collision_mode="geometric_fk",
                )
                if self.planner.set_collision_config(auto_collision):
                    logger.info(
                        "Auto-configured geometric_fk desk safety "
                        "(table_z=%.2f m) from legacy add_table_collision=True. "
                        "Pass CollisionConfig(env_collision_mode='mplib_pointcloud') "
                        "to RobotInterfaceConfig to opt into MPlib point cloud instead.",
                        config.table_z_world,
                    )
                # geometric_fk uses planner's Pinocchio FK — no MPlib point cloud
                use_pointcloud = False

            if use_pointcloud:
                self._setup_table_collision(
                    table_z=config.table_z_world,
                    margin_xy=config.table_margin_xy,
                    n_layers=config.table_layers,
                    layer_spacing=config.table_layer_spacing,
                    xy_resolution=config.table_xy_resolution,
                    x_min_clearance=config.table_x_min_clearance,
                )

        # Hand kinematics
        self.hand_kinematics: HandKinematics | None = None
        if config.hand_urdf_path:
            hk = HandKinematics(config.hand_urdf_path, config.fingertip_link_names or None)
            if hk.is_ready():
                self.hand_kinematics = hk

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def connect(self) -> dict[str, bool]:
        """Connect arm + hand. Returns {"arm": bool, "hand": bool}."""
        result: dict[str, bool] = {}
        result["arm"] = self.arm.connect()
        result["hand"] = self.hand.connect()
        return result

    def disconnect(self) -> None:
        self.arm.disconnect()
        self.hand.disconnect()

    def is_connected(self) -> bool:
        return self.arm.is_connected()

    def check_workspace(self, pos: np.ndarray) -> bool:
        """Check if a 3D position (world frame) is within workspace bounds."""
        return self.workspace.check(pos)

    def check_workspace_orientation(self, eef_quat_wxyz: np.ndarray) -> bool:
        """Check if EEF orientation is within orientation bounds.

        Returns True if orientation_bounds is not configured (backward compatible).
        """
        return self.workspace.check_orientation(eef_quat_wxyz)

    def clamp_workspace_orientation(self, eef_quat_wxyz: np.ndarray) -> np.ndarray:
        """Clamp EEF orientation to orientation bounds.

        Returns input unchanged if orientation_bounds is not configured.
        """
        return self.workspace.clamp_orientation(eef_quat_wxyz)

    def is_error(self) -> bool:
        return self.arm.is_error() or self.hand.is_error()

    def clear_error(self) -> bool:
        arm_ok = self.arm.clear_error()
        hand_ok = self.hand.clear_error()
        return arm_ok and hand_ok

    def emergency_stop(self) -> None:
        self.arm.stop()
        self.hand.stop()

    def get_state(self) -> RobotState:
        """Read arm + hand state with FK computation."""
        arm_state = self.arm.get_state()
        hand_state = self.hand.get_state(full=True)

        arm_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        arm_qvel = np.asarray(arm_state["qvel"], dtype=np.float64)
        arm_tau = np.asarray(arm_state["tau"], dtype=np.float64)

        # EEF FK
        if np.all(np.isfinite(arm_qpos)):
            eef_pose: Pose = self.kinematics.compute_eef_pose_world(arm_qpos)
            eef_pos = eef_pose.p.copy()
            eef_quat_wxyz = eef_pose.q.copy()
            eef_rot6d = quat_wxyz_to_rot6d(eef_quat_wxyz)
        else:
            eef_pos = nan_array(3)
            eef_quat_wxyz = nan_array(4)
            eef_rot6d = nan_array(6)

        hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
        hand_current = np.asarray(hand_state["current"], dtype=np.float64)
        hand_tactile_sum = np.asarray(
            hand_state.get("tactile_force_sum", np.zeros((5, 3))),
            dtype=np.float64,
        )
        hand_tactile_raw = np.asarray(
            hand_state.get("tactile_force_raw", np.zeros((5, 120, 3))),
            dtype=np.float64,
        )
        hand_temperature = np.asarray(
            hand_state.get("temperature", nan_array(12)),
            dtype=np.float64,
        )

        # Fingertip world positions
        fingertip_pos = self._compute_fingertip_pos(
            eef_pos, eef_quat_wxyz, hand_qpos
        )

        hand_error = bool(
            np.any(hand_state.get("commboard_err", np.zeros(12)) != 0)
            or np.any(hand_state.get("jointboard_err", np.zeros(12)) != 0)
            or np.any(hand_state.get("tipboard_err", np.zeros(12)) != 0)
        )

        return RobotState(
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            arm_tau=arm_tau,
            eef_pos=eef_pos,
            eef_quat_wxyz=eef_quat_wxyz,
            eef_rot6d=eef_rot6d,
            hand_qpos=hand_qpos,
            hand_current=hand_current,
            hand_tactile_sum=hand_tactile_sum,
            hand_tactile_raw=hand_tactile_raw,
            hand_temperature=hand_temperature,
            fingertip_pos=fingertip_pos,
            arm_connected=self.arm.is_connected(),
            hand_connected=self.hand.is_connected(),
            hand_error=hand_error,
            timestamp=time.perf_counter(),
        )

    def send_action(self, action: RobotAction) -> dict:
        """Send arm + hand action.

        Returns:
            {"arm_ok": bool, "hand_ok": bool,
             "arm_cmd": ndarray | None,   # (7,) post-clip actual sent value
             "hand_cmd": ndarray | None}  # (12,) post-clip actual sent value

        arm_cmd/hand_cmd are post joint-limit + delta-limit clipped values.
        None on send failure. Recordings should use these post-clip values,
        not the raw IK output.
        """
        result: dict = {}

        arm_ok = self.arm.send_action(action.arm_qpos_cmd)
        hand_ok = self.hand.send_action(action.hand_qpos_cmd)

        result["arm_ok"] = arm_ok
        result["hand_ok"] = hand_ok
        result["arm_cmd"] = self.arm.last_qpos_cmd.copy() if arm_ok else None
        result["hand_cmd"] = self.hand.last_qpos_cmd.copy() if hand_ok else None
        return result

    def reset_soft_start(self) -> None:
        """Reset arm soft-start ramp on TELEOP entry.

        Ensures the soft-start speed ramp always applies to the first
        teleop motion, regardless of idle duration since connect().
        """
        self.arm.reset_soft_start()

    def reset_hand(self) -> bool:
        """Reset hand to home position. Returns False if hand disconnected."""
        if not self.hand.is_connected():
            return False
        return self.hand.reset()

    # ------------------------------------------------------------
    # Return-to-home (split into sub-methods per Phase 3.1)
    # ------------------------------------------------------------

    def return_to_home(
        self,
        use_planning: bool = True,
        cancel_event: object = None,
    ) -> bool:
        """Two-phase return_home: EEF homing → redundant joint homing.

        Phase 1: plan_path(home_eef) — Cartesian path to move EEF to home.
        Phase 2: Joint-space interpolation — current qpos → home_qpos,
                 per-waypoint collision check.

        use_planning=False falls back to direct reset (linear joint space + hand reset).

        SIGINT (Ctrl+C) sets cancel_event to abort waypoint execution.
        """
        with self._install_sigint_handler(cancel_event):
            # 1. Arm not connected → bail out
            if not self.arm.is_connected():
                return False

            # 2. Read current qpos; NaN → fallback
            current_qpos = self._arm_ready()
            if current_qpos is None:
                return self._return_to_home_direct()

            # 3. Not using planning → direct reset
            if not use_planning:
                return self._return_to_home_direct()

            # 4. planner is None → warn + fallback
            if self.planner is None:
                warnings.warn(
                    "use_planning=True but planner is None, falling back to direct reset"
                )
                return self._return_to_home_direct()

            # 5. Get home EEF pose, workspace check/clamp
            home_qpos = self.arm.config.init_qpos.copy()
            home_eef_pose = self._get_home_eef_pose(home_qpos)

            # 5.5 Snap current_qpos to nearest equivalent of home.
            # Continuous joints (J0/J2/J4/J6) can wrap around 2π —
            # e.g. J4 at 155° is physically equivalent to -205°, only
            # 25° from home (-180°).  Without snapping, the raw 335°
            # difference causes IK seeding failures, plan_path rejection,
            # and false "335° from home" error reports.
            current_qpos = self._snap_to_nearest_equivalent(current_qpos, home_qpos)

            # 6. Already at home? (joint error < 1°)
            if self._at_home(current_qpos, home_qpos):
                hand_ok = self.hand.reset() if self.hand.is_connected() else True
                return hand_ok

            dt = float(self.arm.config.dt)

            # ── Phase 0: Hand reset (must happen before plan_path) ──
            current_qpos = self._reset_hand_before_planning(current_qpos)
            if current_qpos is None:
                return self._return_to_home_direct()

            # ── Phase 1: EEF path → home EEF ──
            phase1_completed = self._execute_phase1_eef_cartesian(
                home_eef_pose, current_qpos, dt, cancel_event
            )

            # ── Phase 2: Joint-space homing → home_qpos ──
            if phase1_completed:
                self._execute_phase2_joint_space(home_qpos, dt, cancel_event)
            else:
                # Phase 1 planning failed — typically because the joint-space
                # distance exceeds max_waypoint_delta_deg (default 8°) or the
                # IK solver can't seed from the current equivalent-joint phase.
                # Fall back to SDK joint-space homing: lift EEF for safety,
                # then arm.reset() which uses the SDK's built-in trajectory
                # generator and bypasses planner limitations entirely.
                logger.warning(
                    "Phase 1 plan_path failed, falling back to direct reset "
                    "(SDK joint-space home)"
                )
                return self._return_to_home_direct()

            # ── Verify final distance to home (M2) ──
            final_qpos = self._arm_ready()
            if final_qpos is not None:
                err_rad = self._joint_error_to_home(final_qpos, home_qpos)
                if err_rad > np.deg2rad(10.0):
                    warnings.warn(
                        f"return_to_home incomplete: joint error "
                        f"{np.rad2deg(err_rad):.1f}° from home"
                    )
                    # Large residual error — report failure
                    return False
                if err_rad < _HOME_JOINT_THRESHOLD_RAD:
                    # Already at home — skip redundant hand reset (L1)
                    arm_ok = not self.arm.is_error()
                    return arm_ok

            # Hand reset (degraded if hand not connected)
            hand_ok = self.hand.reset() if self.hand.is_connected() else True
            arm_ok = not self.arm.is_error()
            return arm_ok and hand_ok

    # --- return_to_home sub-methods (Phase 3.1) ---

    @staticmethod
    def _install_sigint_handler(cancel_event: object) -> object:
        """Context manager: install SIGINT handler that sets cancel_event.

        Returns the old handler for restoration.
        """

        class _SigintGuard:
            def __init__(self, event):
                self.event = event
                self.old = None

            def __enter__(self):
                def _on_sigint(signum, frame):
                    if self.event is not None:
                        if hasattr(self.event, "set"):
                            self.event.set()

                self.old = _signal.signal(_signal.SIGINT, _on_sigint)
                return self

            def __exit__(self, *args):
                _signal.signal(_signal.SIGINT, self.old)

        return _SigintGuard(cancel_event)

    def _arm_ready(self) -> np.ndarray | None:
        """Read arm qpos; return None on NaN (caller should fall back)."""
        arm_state = self.arm.get_state()
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        if not np.all(np.isfinite(current_qpos)):
            return None
        return current_qpos

    def _snap_to_nearest_equivalent(
        self, qpos: np.ndarray, reference_qpos: np.ndarray,
    ) -> np.ndarray:
        """Snap continuous joints to the equivalent nearest *reference_qpos*.

        Joints whose range ≥ π (J0/J2/J4/J6 on xArm7) can wrap around 2π
        — e.g. J4 at 155° is physically equivalent to -205°.  Without snapping,
        the raw difference vs home (-180°) is 335°, causing IK seeding failures
        and false distance reports.
        """
        if self.planner is None:
            return qpos
        try:
            return self.planner.ik_mgr.nearest_equivalent_qpos(qpos, reference_qpos)
        except (ValueError, RuntimeError, AttributeError):
            return qpos

    def _joint_error_to_home(
        self, current_qpos: np.ndarray, home_qpos: np.ndarray,
    ) -> float:
        """Max absolute joint error, accounting for equivalent joints."""
        if self.planner is not None:
            try:
                delta = self.planner.ik_mgr.compute_qpos_delta(current_qpos, home_qpos)
                return float(np.max(np.abs(delta)))
            except (ValueError, RuntimeError, AttributeError):
                pass
        return float(np.max(np.abs(current_qpos - home_qpos)))

    @staticmethod
    def _at_home(current_qpos: np.ndarray, home_qpos: np.ndarray) -> bool:
        """Check if all joint errors are < 1° (raw comparison — caller should
        snap to nearest equivalent first via _snap_to_nearest_equivalent)."""
        return float(np.max(np.abs(current_qpos - home_qpos))) < _HOME_JOINT_THRESHOLD_RAD

    def _get_home_eef_pose(self, home_qpos: np.ndarray) -> Pose:
        """FK home qpos → EEF pose, clamp to workspace if needed."""
        home_eef_pose = self.kinematics.compute_eef_pose_world(home_qpos)
        if not self.workspace.check(home_eef_pose.p):
            warnings.warn(
                f"Home EEF position {np.round(home_eef_pose.p, 4)} "
                "is outside workspace, clamping"
            )
            home_eef_pose.p = self.workspace.clamp(home_eef_pose.p)
        return home_eef_pose

    def _reset_hand_before_planning(self, current_qpos: np.ndarray) -> np.ndarray | None:
        """Reset hand to default config before planning.

        The planner URDF model uses the default hand configuration.
        If the real hand is in a non-default pose (e.g. fist/open from teleop),
        desk collision checks will mispredict. Must reset hand first so the
        real configuration matches the planning model.

        Returns updated arm qpos after hand reset (arm may micro-move), or None on NaN.
        """
        if not self.hand.is_connected():
            return current_qpos

        self.hand.reset()
        # Active polling wait for hand convergence (replaces fixed sleep)
        elapsed = 0.0
        while elapsed < _HAND_RESET_CONVERGE_TIMEOUT_S:
            time.sleep(_HAND_RESET_POLL_INTERVAL_S)
            elapsed += _HAND_RESET_POLL_INTERVAL_S
            hand_state = self.hand.get_state()
            hand_qpos = np.asarray(hand_state.get("qpos", []), dtype=np.float64)
            if len(hand_qpos) == 12 and np.all(np.isfinite(hand_qpos)):
                hand_target = np.asarray(self.hand.config.home_qpos, dtype=np.float64)
                if float(np.max(np.abs(hand_qpos - hand_target))) < _HAND_RESET_CONVERGE_THRESHOLD_RAD:
                    break

        # Re-read arm state after hand reset (arm may micro-move)
        arm_state = self.arm.get_state()
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        if not np.all(np.isfinite(current_qpos)):
            return None
        return current_qpos

    def _execute_phase1_eef_cartesian(
        self,
        home_eef_pose: Pose,
        current_qpos: np.ndarray,
        dt: float,
        cancel_event: object,
    ) -> bool:
        """Execute Phase 1: plan_path + dense waypoint execution.

        Returns True if Phase 1 completed successfully.
        """
        try:
            result = self.planner.plan_path(home_eef_pose, current_qpos)
        except RuntimeError as e:
            logger.exception("plan_path failed: %s", e)
            return False

        if not result.success or result.qpos_path is None or len(result.qpos_path) == 0:
            return False

        # Safety: verify fingertips above desk before path execution.
        # This is a last-resort execution-level check; the planner's
        # validate_path() already performed FK desk safety during planning.
        # If fingertips are still below desk here, something went wrong
        # (e.g. hand didn't reset to default pose) — abort execution.
        desk_z: float | None = None
        if self.config.collision is not None:
            desk_z = self.config.collision.table_z_world
        elif self.config.add_table_collision:
            desk_z = self.config.table_z_world

        if desk_z is not None and self.planner is not None:
            hand_state = self.hand.get_state() if self.hand.is_connected() else {"qpos": None}
            actual_hand_qpos = np.asarray(hand_state.get("qpos", []), dtype=np.float64)
            if len(actual_hand_qpos) == 12:
                above_first, min_z_first = self._check_fingertips_above_desk(
                    result.qpos_path[0], actual_hand_qpos, desk_z,
                )
                above_last, min_z_last = self._check_fingertips_above_desk(
                    result.qpos_path[-1], actual_hand_qpos, desk_z,
                )
                if not above_first or not above_last:
                    logger.error(
                        "Phase 1 ABORT: fingertips below desk after hand reset "
                        "(first_z=%.3f m, last_z=%.3f m, desk_z=%.3f m). "
                        "Hand may not have converged to default pose.",
                        min_z_first, min_z_last, desk_z,
                    )
                    return False

        # Dense interpolation (1° step) to avoid _limit_joint_step clipping large jumps
        dense_path = _dense_interpolate(result.qpos_path)
        phase1_completed = True
        for waypoint in dense_path:
            if (cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set()) or self.arm.is_error():
                phase1_completed = False
                break
            if not self._check_arm_torque_ok():
                logger.warning("Phase 1: arm torque exceeded safe limit, aborting waypoint execution")
                phase1_completed = False
                break
            if not self.arm.send_action(waypoint):
                phase1_completed = False
                break
            time.sleep(dt)

        # Closed-loop wait for servo convergence to path endpoint
        if phase1_completed:
            target_qpos = result.qpos_path[-1]
            start_qpos = result.qpos_path[0]
            if not self._wait_for_arm_convergence(target_qpos, start_qpos, dt):
                warnings.warn(
                    "Phase 1 convergence timeout, skipping Phase 2 joint fine-tuning"
                )
                phase1_completed = False

        return phase1_completed

    def _wait_for_arm_convergence(
        self, target_qpos: np.ndarray, start_qpos: np.ndarray | None = None, dt: float = 0.02,
    ) -> bool:
        """Poll arm state until joints converge to *target_qpos*.

        Uses an adaptive timeout: max(dt * multiplier, theoretical_travel_time * multiplier).
        Extracted from _execute_phase1_eef_cartesian (P2.3).
        """
        max_qvel = float(np.min(self.arm.config.max_qvel))
        if start_qpos is not None:
            theoretical_time = float(np.max(np.abs(target_qpos - start_qpos))) / max_qvel
        else:
            theoretical_time = 2.0  # conservative fallback
        max_wait = max(dt * _PHASE1_MAX_WAIT_MULT, theoretical_time * _PHASE1_MAX_WAIT_MULT)
        poll_interval = dt * _PHASE1_POLL_INTERVAL_MULT

        elapsed = 0.0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            try:
                poll_qpos = np.asarray(
                    self.arm.get_state()["qpos"], dtype=np.float64)
                if not np.all(np.isfinite(poll_qpos)):
                    continue
                err = float(np.max(np.abs(poll_qpos - target_qpos)))
                if err < _PHASE1_CONVERGE_THRESHOLD_RAD:
                    return True
            except (ValueError, RuntimeError):
                continue
        return False

    def _execute_phase2_joint_space(
        self,
        home_qpos: np.ndarray,
        dt: float,
        cancel_event: object,
    ) -> None:
        """Execute Phase 2: joint-space interpolation to home_qpos."""
        arm_state = self.arm.get_state()
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        if not np.all(np.isfinite(current_qpos)):
            return

        joint_delta = float(np.max(np.abs(current_qpos - home_qpos)))
        if joint_delta <= _PHASE2_MIN_DELTA_RAD:
            return

        joint_path = self._safe_joint_path(current_qpos, home_qpos)
        if joint_path is not None:
            for waypoint in joint_path:
                if (cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set()) or self.arm.is_error():
                    break
                if not self._check_arm_torque_ok():
                    logger.warning("Phase 2: arm torque exceeded safe limit, aborting waypoint execution")
                    break
                if not self.arm.send_action(waypoint):
                    break
                time.sleep(dt)
        else:
            # Joint path would collide or planner unavailable → skip Phase 2.
            # _return_to_home_direct() uses the same linear joint path
            # (just executed by SDK trajectory generator), which isn't safer.
            # Phase 1 already homed the EEF; skipping Phase 2 only loses
            # exact redundant joint alignment, with no collision risk.
            warnings.warn(
                "Joint-space home path unsafe or unverifiable, "
                "skipping Phase 2 (EEF already at home from Phase 1)"
            )

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _safe_joint_path(
        self, start: np.ndarray, goal: np.ndarray, max_step_rad: float = np.deg2rad(1.0)
    ) -> np.ndarray | None:
        """Linear interpolation start → goal, collision-verified.

        Generates dense path at ``max_step_rad`` resolution for smooth execution,
        then validates with planner.check_path_collisions() which uses 0.02 rad
        internal step size (ref: dimos collision_step_size=0.02).

        Returns None if unsafe or planner unavailable (cannot verify safety).
        """
        dist = float(np.max(np.abs(goal - start)))
        n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
        path = np.array([start + (k / (n - 1)) * (goal - start) for k in range(n)])

        if self.planner is None:
            warnings.warn(
                "_safe_joint_path: planner unavailable, cannot verify collision safety"
            )
            return None

        profile = self.planner.planning_profile
        if profile.check_self_collision:
            result = self.planner.check_path_collisions(path)
            if result.get("path_self_collision"):
                return None
        if profile.check_env_collision:
            result = self.planner.check_path_env_collisions(path)
            if result.get("path_env_collision"):
                return None
        # Geometric FK desk safety (fingertip Z vs table surface).
        # Complements MPlib-based checks above — catches desk collisions even
        # when MPlib point cloud is absent (geometric_fk mode).
        if self.planner.desk_safety is not None and profile.check_env_collision:
            desk_safe, min_z, viol_idx = self.planner.desk_safety.check_path_desk_safety(
                path, step_rad=max_step_rad,
            )
            if not desk_safe:
                logger.warning(
                    "_safe_joint_path: FK desk collision detected "
                    "(fingertip z_min=%.3f m, segment %d)", min_z, viol_idx,
                )
                return None
        return path

    def _check_arm_torque_ok(self) -> bool:
        """Check arm torque is within per-joint safe limits.

        Returns True if torque is safe or unreadable (fail-open: can't check → assume ok).
        """
        arm_state = self.arm.get_state()
        tau = np.asarray(arm_state.get("tau", []), dtype=np.float64)
        if len(tau) != 7 or not np.all(np.isfinite(tau)):
            return True  # can't read, assume ok
        return not np.any(np.abs(tau) >= _ARM_TORQUE_LIMIT_NM)

    def _lift_eef_z_safe(self, current_qpos: np.ndarray) -> bool:
        """Safety lift: move EEF upward to clear desk (C2 fallback).

        Only lifts when the EEF is near the desk surface.  When the EEF is
        already at a safe height, returns True without moving — avoids the
        sudden fast lift that alarms operators during return_home.

        Executes the lift as a dense joint-space interpolation at ~1°/step,
        producing smooth, controlled motion rather than a single high-speed
        servo jump.

        Returns True if EEF is already safe or lift was executed.
        """
        if not np.all(np.isfinite(current_qpos)):
            return False
        current_pose = self.kinematics.compute_eef_pose_world(current_qpos)

        target_z = self._compute_safe_lift_z(float(current_pose.p[2]))
        if target_z is None:
            return True  # already safe, no lift needed

        lift_pose = Pose(
            p=np.array([current_pose.p[0], current_pose.p[1], target_z], dtype=np.float64),
            q=current_pose.q.copy(),
        )
        return self._execute_lift_via_ik(lift_pose, current_qpos)

    def _compute_safe_lift_z(self, current_z: float) -> float | None:
        """Compute a safe target Z for EEF lift.  Returns None if already safe.

        Extracted from _lift_eef_z_safe (P2.3) to separate Z computation
        from motion execution.
        """
        if self.config.collision is not None:
            desk_safe_z = self.config.collision.desk_safe_z
        else:
            desk_safe_z = self.config.table_z_world + _HAND_EXTENSION_Z_M + _HAND_SAFETY_MARGIN_Z_M

        min_safe_z = desk_safe_z + _JOINT_DIP_ALLOWANCE_Z_M
        if current_z >= min_safe_z:
            return None  # already safe

        ws_z_mid = float(np.mean(self.workspace.bounds[2]))
        ws_z_max = float(self.workspace.bounds[2, 1])
        target_z = max(min_safe_z + _Z_LIFT_EXTRA_MARGIN_M, ws_z_mid)
        return min(target_z, ws_z_max - _Z_LIFT_WS_TOP_MARGIN_M)

    def _execute_lift_via_ik(self, lift_pose: Pose, current_qpos: np.ndarray) -> bool:
        """Solve IK for a lift pose and execute as dense joint-space path."""
        if self.planner is None:
            return False

        lift_result = self.planner.solve_teleop_ik(lift_pose, current_qpos, current_qpos)
        if not lift_result.success or lift_result.qpos is None:
            logger.warning(
                "Safety lift: teleop IK failed (%s), falling back to solve_ik",
                lift_result.reason,
            )
            lift_result = self.planner.solve_ik(lift_pose, current_qpos)
        if not lift_result.success or lift_result.qpos is None:
            logger.warning("Safety lift: solve_ik also failed: %s", lift_result.reason)
            return False
        if self.planner.has_env_collision(lift_result.qpos):
            logger.warning("Safety lift result in env-collision, skipping lift")
            return False

        lift_path = _dense_interpolate(
            np.array([current_qpos, lift_result.qpos], dtype=np.float64),
            max_step_rad=np.deg2rad(1.0),
        )
        dt = float(self.arm.config.dt)
        for waypoint in lift_path:
            if not self.arm.send_action(waypoint):
                logger.warning("Safety lift: send_action failed mid-path")
                return False
            time.sleep(dt)
        return True

    def _return_to_home_direct(self) -> bool:
        """Fallback: lift EEF along Z+ away from desk, then joint-space home.

        Linear joint-space motion could pass through the desk.
        Lift EEF first to eliminate collision risk, then run SDK linear trajectory.
        """
        # Phase A: Z+ lift to clear desk
        arm_state = self.arm.get_state()
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        if np.all(np.isfinite(current_qpos)):
            # Snap to nearest equivalent of home so lift IK doesn't
            # reject solutions on spurious "jumps" from wrapped joints.
            home_qpos = self.arm.config.init_qpos.copy()
            current_qpos = self._snap_to_nearest_equivalent(current_qpos, home_qpos)
            if not self._lift_eef_z_safe(current_qpos):
                logger.warning("Direct lift failed or skipped; proceeding to arm.reset()")

        # Phase B: SDK joint-space home
        arm_ok = self.arm.reset()
        hand_ok = self.hand.reset() if self.hand.is_connected() else True
        return arm_ok and hand_ok

    def _setup_table_collision(
        self,
        table_z: float = 0.0,
        margin_xy: float = 0.15,
        n_layers: int = 5,
        layer_spacing: float = 0.01,
        xy_resolution: float = 0.02,
        x_min_clearance: float = 0.15,
    ) -> None:
        """Add a dense point-cloud representation of the table at z=table_z.

        The point cloud covers the workspace footprint plus margin, with
        *n_layers* stacked downward from table_z.  MPlib converts this to
        an octree used by plan_screw / plan_qpos / IK to avoid collisions.

        All coordinates are in the world frame.  x_min_clearance keeps the
        cloud away from the robot base at origin.
        """
        if self.planner is None:
            return

        bounds = self.config.workspace_bounds
        x_min = max(float(bounds[0, 0]), x_min_clearance)
        x_max = float(bounds[0, 1]) + margin_xy
        y_min = float(bounds[1, 0]) - margin_xy
        y_max = float(bounds[1, 1]) + margin_xy

        nx = max(2, int(np.ceil((x_max - x_min) / xy_resolution)) + 1)
        ny = max(2, int(np.ceil((y_max - y_min) / xy_resolution)) + 1)

        xs = np.linspace(x_min, x_max, nx, dtype=np.float64)
        ys = np.linspace(y_min, y_max, ny, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)

        zs = np.linspace(
            table_z, table_z - (n_layers - 1) * layer_spacing, n_layers,
            dtype=np.float64,
        )

        points_list = []
        for z in zs:
            layer = np.column_stack([
                grid_x.ravel(), grid_y.ravel(),
                np.full(grid_x.size, z, dtype=np.float64),
            ])
            points_list.append(layer)

        points = np.vstack(points_list)
        self.planner.add_point_cloud(
            points, name="table", resolution=xy_resolution,
        )
        logger.info(
            "Table collision: %s points, %s layers, z=[%.3f, %.3f] m, "
            "xy=[%.2f,%.2f]x[%.2f,%.2f] m",
            points.shape[0], n_layers, zs[-1], zs[0],
            x_min, x_max, y_min, y_max,
        )

    def _check_fingertips_above_desk(
        self, arm_qpos: np.ndarray, hand_qpos: np.ndarray, desk_z: float = 0.0,
    ) -> tuple[bool, float]:
        """Check if fingertips are above desk using actual hand_qpos + arm waypoint FK.

        Returns: (all_above, min_z). Only used for execution-level validation
        when hand_kinematics is available.
        """
        if self.hand_kinematics is None or not self.hand_kinematics.is_ready():
            return True, float("inf")

        if not np.all(np.isfinite(arm_qpos)) or not np.all(np.isfinite(hand_qpos)):
            return True, float("inf")

        eef = self.kinematics.compute_eef_pose_world(arm_qpos)
        tips = self._compute_fingertip_pos(eef.p, eef.q, hand_qpos)
        if not np.all(np.isfinite(tips)):
            return True, float("inf")

        min_z = float(np.min(tips[:, 2]))
        return min_z >= desk_z, min_z

    def _compute_fingertip_pos(
        self,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
        hand_qpos: np.ndarray,
    ) -> np.ndarray:
        if self.hand_kinematics is None or not self.hand_kinematics.is_ready():
            return nan_array((5, 3))

        if not np.all(np.isfinite(eef_pos)) or not np.all(np.isfinite(hand_qpos)):
            return nan_array((5, 3))

        tips_in_handbase = self.hand_kinematics.compute_tip_positions_in_handbase(hand_qpos)
        if not np.all(np.isfinite(tips_in_handbase)):
            return nan_array((5, 3))

        T_world_eef = Pose(p=eef_pos, q=eef_quat_wxyz)

        T_eef_handbase = Pose(
            p=self.config.T_eef_handbase_pos,
            q=self.config.T_eef_handbase_quat_wxyz,
        )

        tips_world = np.zeros((5, 3), dtype=np.float64)
        for i in range(5):
            tip_in_handbase = Pose(
                p=tips_in_handbase[i],
                q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            T_world_tip = compose_pose(
                compose_pose(T_world_eef, T_eef_handbase),
                tip_in_handbase,
            )
            tips_world[i] = T_world_tip.p

        return tips_world

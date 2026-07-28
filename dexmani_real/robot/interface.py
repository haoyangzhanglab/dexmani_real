"""RobotInterface — arm + hand unified interface.

Controllers operate hardware exclusively through RobotInterface, never calling
XArm7/XHand directly.

Arm position servo is handled by ArmInnerLoop (in-process 30Hz daemon thread, mode 6).
interface.py handles hand commands and blocking arm moves (reset, home).
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from dexmani_real.planning import WorkspaceSafety
from dexmani_real.planning.kinematics import XArm7Kinematics
from dexmani_real.planning.pose_utils import compose_pose, quat_wxyz_to_rot6d
from dexmani_real.planning.types import Pose
from dexmani_real.robot.hand_kinematics import HandKinematics
from dexmani_real.robot.hand_process import make_hand_servo
from dexmani_real.robot.types import RobotAction, RobotInterfaceConfig, RobotState
from dexmani_real.robot.xarm7 import XArm7

if TYPE_CHECKING:
    from dexmani_real.robot.arm_process import ArmServo
from dexmani_real.robot.xhand import XHand, XHandConfig
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.robot.hand_process import HandSHMAdapter

logger = get_logger(__name__)


class RobotInterface:
    """Arm + Hand unified interface — hand commands + blocking arm moves.

    Teleop arm position servo is handled by ArmInnerLoop (in-process daemon thread).
    This class manages:
      - Hand send_action (12-DOF position commands)
      - Arm blocking moves (reset, emergency stop)
      - State reading (arm + hand + FK)
      - Workspace safety checks
    """

    def __init__(
        self,
        config: RobotInterfaceConfig,
        kinematics: XArm7Kinematics,
        *,
        planner: XArm7MotionPlanner | None = None,
        hand_factory: Callable[[XHandConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self.kinematics = kinematics
        self.planner = planner
        self.workspace = WorkspaceSafety(config.workspace_bounds)

        self.arm = XArm7(config.arm)
        self._arm_servo: ArmServo | None = None  # registered by entry point via set_arm_servo()
        # Hand: in-process XHand (today) or crash-isolated HandSHMAdapter
        # subprocess when the hand transition flag is on (plan §6 P1). Both
        # satisfy the XHand duck-type this class + validate_action use, so no
        # hand call site changes. hand_factory: test seam (no hardware).
        self.hand: XHand | HandSHMAdapter = make_hand_servo(
            config.hand,
            hand_factory=hand_factory,
        )

        # Validate home EEF is within workspace
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

        # Table collision geometry — lightweight CollisionModel (no MPlib point cloud penalty)
        if self.planner is not None and config.collision is not None:
            if config.collision.enable_env_collision:
                try:
                    self.planner.collision_model.add_table(
                        table_height=config.collision.table_z_world,
                        x_center=config.collision.table_x_center,
                        half_x=config.collision.table_half_x,
                        half_y=config.collision.table_half_y,
                        half_z=config.collision.table_half_z,
                    )
                except RuntimeError:
                    # FCL bindings unavailable — Tier-2 env collision disabled.
                    # Desk safety is still fully covered by FingertipDeskSafety (FK-based).
                    logger.warning(
                        "Cannot register table obstacle: FCL bindings unavailable. "
                        "Tier-2 FCL env collision disabled. "
                        "Desk safety covered by FingertipDeskSafety (FK-based)."
                    )

        # Hand kinematics
        self.hand_kinematics: HandKinematics | None = None
        if config.hand_urdf_path:
            hk = HandKinematics(config.hand_urdf_path, config.fingertip_link_names or None)
            if hk.is_ready():
                self.hand_kinematics = hk

    # ── Lifecycle ──

    def connect(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        try:
            result["arm"] = self.arm.connect()
        except Exception as e:
            logger.warning("Arm connect() raised exception: %s", e)
            result["arm"] = False
        try:
            result["hand"] = self.hand.connect()
        except Exception as e:
            logger.warning("Hand connect() raised exception: %s", e)
            result["hand"] = False
        return result

    def disconnect(self) -> None:
        try:
            self.arm.disconnect()
        except Exception as e:
            logger.warning("Arm disconnect() exception: %s", e)
        try:
            self.hand.disconnect()
        except Exception as e:
            logger.warning("Hand disconnect() exception: %s", e)

    def is_connected(self) -> bool:
        return self.arm.is_connected()

    def check_workspace(self, pos: np.ndarray) -> bool:
        return self.workspace.check(pos)

    def clamp_workspace_pos(self, pos: np.ndarray) -> np.ndarray:
        return self.workspace.clamp(np.asarray(pos, dtype=np.float64))

    def is_error(self) -> bool:
        return self.arm.is_error() or self.hand.is_error()

    def clear_error(self) -> bool:
        try:
            arm_ok = self.arm.clear_error()
        except Exception as e:
            logger.warning("Arm clear_error() exception: %s", e)
            arm_ok = False
        try:
            hand_ok = self.hand.clear_error()
        except Exception as e:
            logger.warning("Hand clear_error() exception: %s", e)
            hand_ok = False
        return arm_ok and hand_ok

    def set_arm_servo(self, servo: ArmServo) -> None:
        """Register the position servo so :meth:`emergency_stop` can coordinate it."""
        self._arm_servo = servo

    def emergency_stop(self) -> None:
        # Fast-path estop through the isolated servo first (≤1 tick),
        # then fall back to direct SDK stop for the blocking connection.
        if self._arm_servo is not None:
            try:
                self._arm_servo.emergency_stop()
            except Exception as e:
                logger.warning("ArmServo emergency_stop() exception: %s", e)
        try:
            self.arm.stop()
        except Exception as e:
            logger.warning("Arm emergency_stop() exception: %s", e)
        try:
            self.hand.stop()
        except Exception as e:
            logger.warning("Hand emergency_stop() exception: %s", e)

    # ── State ──

    def get_state(
        self,
        arm_qpos: np.ndarray | None = None,
        arm_qvel: np.ndarray | None = None,
        arm_tau: np.ndarray | None = None,
    ) -> RobotState:
        """Read arm + hand state with FK computation.

        Args:
            arm_qpos: Optional arm joint positions from ArmInnerLoop.get_state().
                      When provided, skips arm SDK call (used in teleop loop).
            arm_qvel: Optional joint velocities from ArmInnerLoop.get_dynamics().
                      Only used together with arm_qpos; NaN placeholder if omitted.
            arm_tau: Optional joint torques from ArmInnerLoop.get_dynamics().
                     Only used together with arm_qpos; NaN placeholder if omitted.
        """
        if arm_qpos is not None:
            arm_qvel = nan_array(7) if arm_qvel is None else np.asarray(arm_qvel, dtype=np.float64)
            arm_tau = nan_array(7) if arm_tau is None else np.asarray(arm_tau, dtype=np.float64)
        else:
            try:
                arm_state = self.arm.get_state()
                arm_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
                arm_qvel = np.asarray(arm_state["qvel"], dtype=np.float64)
                arm_tau = np.asarray(arm_state["tau"], dtype=np.float64)
            except Exception:
                arm_qpos = nan_array(7)
                arm_qvel = nan_array(7)
                arm_tau = nan_array(7)

        try:
            hand_state = self.hand.get_state()  # default mode now includes tactile (ref: DexUMI)
            hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
            hand_current = np.asarray(hand_state.get("current", np.zeros(12)), dtype=np.float64)
            hand_tactile_sum = np.asarray(hand_state.get("tactile_force_sum", np.zeros((5, 3))), dtype=np.float64)
            hand_tactile_force = np.asarray(hand_state.get("tactile_force", np.zeros((5, 120, 3))), dtype=np.float64)
            hand_tactile_contact = np.asarray(hand_state.get("tactile_contact", np.zeros(5, dtype=bool)), dtype=bool)
            hand_tipboard_err = np.asarray(hand_state.get("tipboard_err", np.zeros(12, dtype=np.int32)), dtype=np.int32)
        except Exception:
            hand_qpos = nan_array(12)
            hand_current = nan_array(12)
            hand_tactile_sum = nan_array((5, 3))
            hand_tactile_force = nan_array((5, 120, 3))
            hand_tactile_contact = np.zeros(5, dtype=bool)
            hand_tipboard_err = np.zeros(12, dtype=np.int32)

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

        fingertip_pos = self._compute_fingertip_pos(eef_pos, eef_quat_wxyz, hand_qpos)

        return RobotState(
            arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
            arm_qvel=arm_qvel,
            arm_tau=arm_tau,
            eef_pos=eef_pos,
            eef_quat_wxyz=eef_quat_wxyz,
            eef_rot6d=eef_rot6d,
            hand_qpos=hand_qpos,
            hand_current=hand_current,
            hand_tactile_sum=hand_tactile_sum,
            hand_tactile_force=hand_tactile_force,
            hand_tactile_contact=hand_tactile_contact,
            hand_tipboard_err=hand_tipboard_err,
            fingertip_pos=fingertip_pos,
            arm_connected=self.arm.is_connected(),
            hand_connected=self.hand.is_connected(),
            timestamp=time.perf_counter(),
        )

    # ── Hand action ──

    def send_action(self, action: RobotAction) -> dict:
        """Send hand action only (arm is handled by ArmInnerLoop).

        Returns:
            {"hand_ok": bool, "hand_cmd": ndarray | None}
        """
        hand_ok = self.hand.send_action(action.hand_qpos_cmd)
        return {
            "hand_ok": hand_ok,
            "hand_cmd": self.hand.last_qpos_cmd.copy() if (hand_ok and self.hand.last_qpos_cmd is not None) else None,
        }

    # ── Hand reset ──

    def reset_hand(self) -> bool:
        if not self.hand.is_connected():
            return False
        return self.hand.reset()

    def _sync_hand_collision_model(self) -> None:
        """Sync CollisionModel hand buffer with current hardware state.

        Non-critical: CollisionModel defaults to open hand on failure.
        The 19-DOF full URDF (xarm7_xhand_right.urdf) includes active hand
        joints, so hand pose affects all collision checks.  Keeping this
        buffer current avoids false-positive env collisions when the hand
        is near the table.
        """
        if self.planner is None:
            return
        try:
            hand_state = self.hand.get_state()
            hand_qpos = np.asarray(hand_state.get("qpos", np.zeros(12)), dtype=np.float64)
            if hand_qpos.shape == (12,) and np.all(np.isfinite(hand_qpos)):
                self.planner.set_hand_qpos(hand_qpos)
        except Exception:
            pass  # non-critical

    # ── Return to home (path-planned) ──

    def return_to_home(self, *, home_dt: float | None = None) -> bool:
        """Path-planned return-to-home with collision avoidance.

        Three-tier execution (in priority order):
          Tier 1: plan_path(home EEF) — screw/RRT Cartesian path with full
                  collision checking (self + env + desk + workspace).
          Tier 2: Safe joint-space interpolation — dense linear joint-space
                  path at 1° resolution, collision-checked. Used when
                  plan_path fails (e.g. waypoint delta too large).
          Tier 3: arm.reset() — SDK raw blocking move, NO collision avoidance.
                  Only used when both Tier 1 and Tier 2 are unavailable or fail.

        After Cartesian/joint approach, a final arm.reset() is always called
        for sub-degree convergence to exact init_qpos.

        Args:
            home_dt: Sleep interval between waypoints (s). Default: arm.config.dt
                     (~0.02s → ~50°/s). Increase for slower, safer homing
                     (e.g. 0.04 → ~25°/s).
        """
        if not self.arm.is_connected():
            return False

        home_qpos = self.arm.config.init_qpos.copy()
        dt = home_dt if home_dt is not None else float(self.arm.config.dt)

        # ── 1. Read current position ──
        qpos = self._read_arm_qpos()
        if qpos is None:
            return self._reset_blocking()

        # ── 2. No planner → safe joint fallback → reset ──
        if self.planner is None:
            logger.warning("No planner available, trying safe joint fallback")
            if not self._safe_joint_home_fallback(qpos, home_qpos, dt):
                return self._reset_blocking()
            # Continue to Phase 2 + final reset below
            qpos = self._read_arm_qpos()
            if qpos is None:
                return self._reset_blocking()

        else:
            # ── 3. Snap continuous joints (J0/J2/J4/J6) to nearest 2π-equivalent ──
            try:
                qpos = self.planner.nearest_equivalent_qpos(qpos, home_qpos)
            except Exception:
                pass

            # ── 4. Already at home? ──
            if float(np.max(np.abs(qpos - home_qpos))) < np.deg2rad(1.0):
                hand_ok = self.hand.reset() if self.hand.is_connected() else True
                # Sync CollisionModel with post-reset hand state
                self._sync_hand_collision_model()
                return hand_ok

            # ── 5. Pre-flight: reset hand (align FK model to reality) ──
            if self.hand.is_connected():
                self.hand.reset()
                time.sleep(0.3)
                # Sync CollisionModel hand buffer so env collision checks use
                # the post-reset hand geometry (defence-in-depth; today the
                # 7-DOF collision URDF ignores hand DOFs entirely).
                self._sync_hand_collision_model()
            qpos = self._read_arm_qpos()
            if qpos is None:
                return self._reset_blocking()

            # ── 6. Tier 1: Plan + execute EEF Cartesian path ──
            home_eef = self.kinematics.compute_eef_pose_world(home_qpos)
            plan_ok = False
            plan_reason = ""
            try:
                result = self.planner.plan_path(home_eef, qpos)
                if result.success and result.qpos_path is not None and len(result.qpos_path) > 0:
                    plan_ok = True
                    logger.info(
                        "return_to_home Phase 1: %d waypoints, source=%s, score=%.3f",
                        len(result.qpos_path),
                        result.source,
                        result.report.get("path_score", float("nan")),
                    )
                    if not self._execute_waypoints(result.qpos_path, dt):
                        logger.warning(
                            "return_to_home Phase 1 execution aborted mid-path: %s",
                            self.arm.last_error_message,
                        )
                else:
                    plan_reason = result.reason or "unknown"
            except Exception:
                logger.warning("plan_path exception", exc_info=True)
                plan_reason = "exception"

            if not plan_ok:
                # ── Tier 2: Safe joint-space fallback (collision-checked) ──
                logger.info(
                    "plan_path failed: %s, trying safe joint-space fallback",
                    plan_reason,
                )
                if not self.arm.is_error() and not self._safe_joint_home_fallback(qpos, home_qpos, dt):
                    logger.warning("Safe joint fallback also failed, falling back to arm.reset()")
                    return self._reset_blocking()

        # ── 7. Phase 2: Joint-space interpolation to exact home ──
        if not self.arm.is_error():
            qpos = self._read_arm_qpos()
            if qpos is not None:
                self._execute_joint_homing(qpos, home_qpos, dt)

        # ── 8. Finalize: blocking converge to exact init_qpos ──
        # Phase 2's send_action() uses non-blocking set_servo_angle_j() —
        # the arm may not have physically settled.  arm.reset() calls
        # set_servo_angle(wait=True) which blocks until the arm reaches
        # init_qpos within the SDK's internal convergence tolerance.
        if not self.arm.is_error():
            arm_ok = self.arm.reset()
            if not arm_ok:
                logger.warning("Blocking reset failed: %s", self.arm.last_error_message)
        else:
            arm_ok = False

        hand_ok = self.hand.reset() if self.hand.is_connected() else True
        # Re-read actual position to verify precision
        final = self._read_arm_qpos()
        if final is not None:
            err_deg = float(np.rad2deg(np.max(np.abs(final - home_qpos))))
            logger.info("return_to_home final error: %.2f deg", err_deg)
        return arm_ok and hand_ok

    # ── Return-to-home helpers ──

    def _read_arm_qpos(self) -> np.ndarray | None:
        """Read arm joint positions; return None on NaN/error."""
        try:
            arm_state = self.arm.get_state()
            qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
            if np.all(np.isfinite(qpos)):
                return qpos
        except Exception:
            pass
        return None

    def _reset_blocking(self) -> bool:
        """Fallback: SDK blocking move — no collision avoidance."""
        arm_ok = self.arm.reset()
        hand_ok = self.hand.reset() if self.hand.is_connected() else True
        return arm_ok and hand_ok

    def _execute_waypoints(self, path: np.ndarray, dt: float, max_step_rad: float = np.deg2rad(1.0)) -> bool:
        """Execute a joint-space path via position servo with dense interpolation.

        Densely interpolates waypoints at ``max_step_rad`` resolution (default 1°)
        for smooth motion without large joint-step clipping. Aborts on arm error.
        """
        if len(path) == 0:
            return True
        if len(path) == 1:
            return self.arm.send_action(path[0]) if not self.arm.is_error() else False

        # Build dense path: linear interpolation at max_step_rad resolution
        dense: list[np.ndarray] = [path[0]]
        for i in range(len(path) - 1):
            seg_dist = float(np.max(np.abs(path[i + 1] - path[i])))
            n = max(1, int(np.ceil(seg_dist / max_step_rad)))
            for k in range(1, n + 1):
                alpha = k / n
                dense.append(path[i] + alpha * (path[i + 1] - path[i]))

        for waypoint in dense:
            if self.arm.is_error():
                return False
            if not self.arm.send_action(waypoint):
                return False
            time.sleep(dt)
        return True

    def _check_joint_path_safe(self, path: np.ndarray) -> bool:
        """Check self-collision + env-collision + FK desk safety for a joint path.

        Returns True if all checks pass or planner unavailable (can't verify).
        """
        if self.planner is None:
            return True  # can't verify, caller decides

        profile = self.planner.planning_profile
        if profile.check_self_collision:
            result = self.planner.check_path_collisions(path)
            if result.get("path_self_collision"):
                return False
        if profile.check_env_collision:
            result = self.planner.check_path_env_collisions(path)
            if result.get("path_env_collision"):
                return False
        if profile.check_env_collision and self.planner.desk_safety is not None:
            desk_safe, _min_z, _idx = self.planner.desk_safety.check_path_desk_safety(path)
            if not desk_safe:
                return False
        return True

    def _execute_joint_homing(self, current: np.ndarray, target: np.ndarray, dt: float) -> None:
        """Phase 2: collision-checked joint-space interpolation to exact home.

        Skips if joint delta is negligible or the linear path has collisions.
        """
        delta = float(np.max(np.abs(current - target)))
        if delta < np.deg2rad(0.5):
            return

        n = max(2, int(np.ceil(delta / np.deg2rad(1.0))) + 1)
        path = np.array(
            [current + (k / (n - 1)) * (target - current) for k in range(n)],
            dtype=np.float64,
        )

        if not self._check_joint_path_safe(path):
            logger.warning("Phase 2 joint path has collisions, skipping " "(EEF already at home from Phase 1)")
            return

        logger.info("return_to_home Phase 2: %d joint waypoints, delta=%.1f°", n, np.rad2deg(delta))
        if not self._execute_waypoints(path, dt):
            logger.warning(
                "return_to_home Phase 2 execution aborted mid-path: %s",
                self.arm.last_error_message,
            )

    def _safe_joint_home_fallback(self, current: np.ndarray, target: np.ndarray, dt: float) -> bool:
        """Tier 2 fallback: collision-checked joint-space interpolation to home.

        Used when plan_path fails (e.g. waypoint delta too large after shortcut
        smoothing). Builds a dense 1°-resolution joint-space path, checks
        self/env/desk collisions, and executes if safe.

        When the direct linear path has collisions (common when the arm is
        stretched out near the table), retries with a two-stage detour:
          1. Move base/shoulder/elbow (J0-J2) to home first → lifts arm clear
          2. Then move wrist joints (J3-J6) to home

        Returns:
            True if arm is already close enough or the path was safe and executed.
            False if collisions were detected or execution failed.
        """
        delta = float(np.max(np.abs(current - target)))
        if delta < np.deg2rad(0.5):
            return True

        # ── Attempt 1: direct linear interpolation ──
        n = max(2, int(np.ceil(delta / np.deg2rad(1.0))) + 1)
        path = np.array(
            [current + (k / (n - 1)) * (target - current) for k in range(n)],
            dtype=np.float64,
        )

        if self._check_joint_path_safe(path):
            logger.info(
                "return_to_home safe joint fallback: %d waypoints, delta=%.1f°",
                n,
                np.rad2deg(delta),
            )
            return self._execute_waypoints(path, dt)

        # ── Attempt 2: two-stage detour (lift arm structure first) ──
        # J0=base, J1=shoulder, J2=elbow → move to home first (lifts arm clear of table)
        # J3-J6=wrist → keep current during stage 1, move to home in stage 2
        PROXIMAL_MASK = np.array([True, True, True, False, False, False, False], dtype=bool)

        if np.any(PROXIMAL_MASK):
            mid = current.copy()
            mid[PROXIMAL_MASK] = target[PROXIMAL_MASK]

            # Stage 1: proximal joints → home (wrist stays)
            delta1 = float(np.max(np.abs(mid - current)))
            n1 = max(2, int(np.ceil(delta1 / np.deg2rad(1.0))) + 1)
            path1 = np.array(
                [current + (k / (n1 - 1)) * (mid - current) for k in range(n1)],
                dtype=np.float64,
            )

            # Stage 2: wrist joints → home
            delta2 = float(np.max(np.abs(target - mid)))
            n2 = max(2, int(np.ceil(delta2 / np.deg2rad(1.0))) + 1)
            path2 = np.array(
                [mid + (k / (n2 - 1)) * (target - mid) for k in range(1, n2)],  # skip mid (already at end of path1)
                dtype=np.float64,
            )

            staged_path = np.concatenate([path1, path2], axis=0) if len(path2) > 0 else path1

            if self._check_joint_path_safe(staged_path):
                logger.info(
                    "return_to_home safe joint fallback (2-stage): "
                    "stage1=%d wp (proximal), stage2=%d wp (wrist), total_delta=%.1f°",
                    n1,
                    n2 + 1,
                    np.rad2deg(delta),
                )
                return self._execute_waypoints(staged_path, dt)

        logger.warning(
            "Safe joint fallback: path has collisions (self/env/desk), delta=%.1f°",
            np.rad2deg(delta),
        )
        return False

    # ── Fingertip FK ──

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
        T_eef_handbase = Pose(p=self.config.T_eef_handbase_pos, q=self.config.T_eef_handbase_quat_wxyz)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        tips_world = np.zeros((5, 3), dtype=np.float64)
        for i in range(5):
            tip_in_handbase = Pose(p=tips_in_handbase[i], q=identity_quat)
            T_world_tip = compose_pose(compose_pose(T_world_eef, T_eef_handbase), tip_in_handbase)
            tips_world[i] = T_world_tip.p

        return tips_world

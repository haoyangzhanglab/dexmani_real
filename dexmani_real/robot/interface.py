"""RobotInterface — arm + hand unified interface.

Controllers operate hardware exclusively through RobotInterface, never calling
XArm7/XHand directly.

Arm position servo is handled by ArmInnerLoop (in-process 30Hz daemon thread, mode 6).
interface.py handles hand commands and blocking arm moves (reset, home).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from dexmani_real.planning import WorkspaceSafety
from dexmani_real.planning.kinematics import XArm7Kinematics
from dexmani_real.planning.path_utils import interpolate_waypoints
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
                logger.warning(f"Cannot validate home EEF workspace (NaN FK): {msg}")

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
                logger.warning("arm get_state failed", exc_info=True)

        try:
            hand_state = self.hand.get_state()  # default mode now includes tactile (ref: DexUMI)
            hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
            hand_current = np.asarray(hand_state.get("current", np.zeros(12)), dtype=np.float64)
            hand_tactile_sum = np.asarray(hand_state.get("tactile_force_sum", np.zeros((5, 3))), dtype=np.float64)
            hand_tactile_force = np.asarray(hand_state.get("tactile_force", np.zeros((5, 120, 3))), dtype=np.float64)
            hand_tactile_contact = np.asarray(hand_state.get("tactile_contact", np.zeros(5, dtype=bool)), dtype=bool)
            hand_tipboard_err = np.asarray(hand_state.get("tipboard_err", np.zeros(12, dtype=np.int32)), dtype=np.int32)
            hand_commboard_err = np.asarray(hand_state.get("commboard_err", np.zeros(12, dtype=np.int32)), dtype=np.int32)
            hand_jointboard_err = np.asarray(hand_state.get("jointboard_err", np.zeros(12, dtype=np.int32)), dtype=np.int32)
        except Exception:
            hand_qpos = nan_array(12)
            hand_current = nan_array(12)
            hand_tactile_sum = nan_array((5, 3))
            hand_tactile_force = nan_array((5, 120, 3))
            hand_tactile_contact = np.zeros(5, dtype=bool)
            hand_tipboard_err = np.zeros(12, dtype=np.int32)
            hand_commboard_err = np.zeros(12, dtype=np.int32)
            hand_jointboard_err = np.zeros(12, dtype=np.int32)
            logger.warning("hand get_state failed", exc_info=True)

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
            hand_commboard_err=hand_commboard_err,
            hand_jointboard_err=hand_jointboard_err,
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

        Only active in 19-DOF mode where the collision URDF includes hand DOFs.
        In 7-DOF mode (current production), the URDF ignores hand DOFs entirely.
        """
        if self.planner is None:
            return
        if not self.planner.collision_model.hand_dof:
            return
        try:
            hand_state = self.hand.get_state()
            hand_qpos = np.asarray(hand_state.get("qpos", np.zeros(12)), dtype=np.float64)
            if hand_qpos.shape == (12,) and np.all(np.isfinite(hand_qpos)):
                self.planner.set_hand_qpos(hand_qpos)
        except Exception:
            pass  # non-critical; hand FK model is defence-in-depth only

    # ── Return to home (path-planned) ──

    def _try_cartesian_home(
        self, qpos: np.ndarray, home_qpos: np.ndarray, home_eef: "Pose", dt: float
    ) -> bool:
        """Tier 1: plan + execute a cartesian path to home EEF pose.

        Returns:
            True if a path was planned (execution is best-effort — partial
            execution does not fail this method). False if plan_path itself
            returned no valid path, so the caller should escalate to Tier 2.
        """
        assert self.planner is not None  # caller guarantees planner is set
        try:
            result = self.planner.plan_path(home_eef, qpos)
            if result.success and result.qpos_path is not None and len(result.qpos_path) > 0:
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
                # Path was planned — Tier 1 is "ok" even if execution aborted.
                return True
            return False
        except Exception:
            logger.warning("plan_path exception", exc_info=True)
            return False

    def return_to_home(self, *, home_dt: float | None = None) -> bool:
        """Coarse return-to-home with collision avoidance.

        Three-tier execution (in priority order):
          Tier 1: plan_path(home EEF) — screw/RRT Cartesian path with full
                  collision checking (self + env + workspace).
          Tier 2: Safe joint-space interpolation — dense linear joint-space
                  path at 1° resolution, collision-checked. Used when
                  plan_path fails (e.g. waypoint delta too large).
          Tier 3: arm.reset() — SDK raw blocking move, NO collision avoidance.
                  Only used when both Tier 1 and Tier 2 are unavailable or fail.

        After Tier 1, the EEF is at the home pose but the joint configuration
        may differ from the canonical home_qpos (nullspace ambiguity of a
        7-DOF arm).  This joint-space residual is handled by the caller
        (do_return_home) via Mode 6 firmware trajectory planning — much
        smoother than linear joint interpolation which causes EEF wobble.

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
            # _safe_joint_home_fallback performs its own joint-space interpolation;
            # fine convergence is handled by the caller via Mode 6.
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
            if not self._try_cartesian_home(qpos, home_qpos, home_eef, dt):
                # ── Tier 2: Safe joint-space fallback (collision-checked) ──
                logger.info("plan_path failed, trying safe joint-space fallback")
                if not self.arm.is_error() and not self._safe_joint_home_fallback(qpos, home_qpos, dt):
                    logger.warning("Safe joint fallback also failed, falling back to arm.reset()")
                    return self._reset_blocking()
            else:
                # Phase 1 (Cartesian path) succeeded — the EEF is at home.
                # The joint-space residual (nullspace difference between the
                # IK solution that plan_path chose and the canonical home_qpos)
                # can be large for a 7-DOF arm (typically 5-40°).  When the
                # residual exceeds the threshold where Mode 6 convergence is
                # guaranteed smooth, run a collision-checked joint-space
                # interpolation (Phase 2) to reduce it.
                #
                # Below JOINT_RESIDUAL_THRESHOLD_DEG (5°), Mode 6 convergence
                # with the per-step delta clamp (0.15 rad ≈ 8.6°/step) handles
                # the residual smoothly.  Above 5°, the residual is large
                # enough that Mode 6 takes multiple seconds to converge, and
                # the initial steps risk firmware overspeed trips (C24).
                # Running Phase 2 here reduces the residual to sub-degree
                # before the inner loop restart, making the Mode 6 convergence
                # step nearly instantaneous.
                #
                # The EEF wobble concern (linear joint interpolation between
                # two configurations with identical EEF poses) is real but
                # bounded: for residuals ≤ 40° at 1° interpolation, the
                # maximum EEF deviation is < 2 cm, entirely within the safe
                # workspace around the home pose.
                JOINT_RESIDUAL_THRESHOLD_DEG = 5.0
                qpos_after_phase1 = self._read_arm_qpos()
                if qpos_after_phase1 is not None and not self.arm.is_error():
                    residual_deg = float(np.rad2deg(np.max(np.abs(qpos_after_phase1 - home_qpos))))
                    if residual_deg > JOINT_RESIDUAL_THRESHOLD_DEG:
                        logger.info(
                            "return_to_home Phase 1 residual %.1f° > %.1f° — "
                            "running joint-space pre-convergence (Phase 2)",
                            residual_deg,
                            JOINT_RESIDUAL_THRESHOLD_DEG,
                        )
                        self._execute_joint_homing(qpos_after_phase1, home_qpos, dt)

        # ── 7. Coarse approach complete — log final error for observability ──
        # Fine convergence to exact home_qpos is handled by the caller
        # (do_return_home) via Mode 6 inner-loop smooth convergence.

        # Settling delay: _execute_waypoints sends non-blocking
        # set_servo_angle_j(wait=False) commands, so the arm may still be
        # in motion when this read fires.  A brief dwell ensures the
        # reported coarse error reflects the true endpoint, giving the
        # caller's Mode 6 convergence loop a more accurate starting point.
        time.sleep(0.3)

        final = self._read_arm_qpos()
        if final is not None:
            err_deg = float(np.rad2deg(np.max(np.abs(final - home_qpos))))
            logger.info("return_to_home coarse approach final error: %.2f deg", err_deg)
        return True

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
        # Best-effort: caller falls back to _reset_blocking() on None
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

        dense = interpolate_waypoints(path, max_step_rad)
        for waypoint in dense:
            if self.arm.is_error():
                return False
            if not self.arm.send_action(waypoint):
                return False
            time.sleep(dt)
        return True

    def _check_joint_path_safe(self, path: np.ndarray) -> bool:
        """Check self-collision + env-collision for a joint path.

        Returns True if all checks pass or planner unavailable (can't verify).
        """
        if self.planner is None:
            return True  # can't verify, caller decides

        profile = self.planner.planning_profile
        if profile.check_self_collision:
            result = self.planner.check_path_collisions(path)
            if result.get("path_self_collision"):
                return False
        return True

    def _execute_joint_homing(self, current: np.ndarray, target: np.ndarray, dt: float) -> None:
        """Phase 2: collision-checked joint-space interpolation to exact home.

        Skips if joint delta is negligible or the linear path has collisions.
        """
        delta = float(np.max(np.abs(current - target)))
        if delta < np.deg2rad(0.5):
            return

        path = interpolate_waypoints(np.stack([current, target]), np.deg2rad(1.0))

        if not self._check_joint_path_safe(path):
            logger.warning("Phase 2 joint path has collisions, skipping " "(EEF already at home from Phase 1)")
            return

        logger.info("return_to_home Phase 2: %d joint waypoints, delta=%.1f°", len(path), np.rad2deg(delta))
        if not self._execute_waypoints(path, dt):
            logger.warning(
                "return_to_home Phase 2 execution aborted mid-path: %s",
                self.arm.last_error_message,
            )

    def _safe_joint_home_fallback(self, current: np.ndarray, target: np.ndarray, dt: float) -> bool:
        """Tier 2 fallback: collision-checked joint-space interpolation to home.

        Used when plan_path fails (e.g. waypoint delta too large after shortcut
        smoothing). Builds a dense 1°-resolution joint-space path, checks
        self/env collisions, and executes if safe.

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
        path = interpolate_waypoints(np.stack([current, target]), np.deg2rad(1.0))

        if self._check_joint_path_safe(path):
            logger.info(
                "return_to_home safe joint fallback: %d waypoints, delta=%.1f°",
                len(path),
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
            path1 = interpolate_waypoints(np.stack([current, mid]), np.deg2rad(1.0))

            # Stage 2: wrist joints → home
            path2_full = interpolate_waypoints(np.stack([mid, target]), np.deg2rad(1.0))
            path2 = path2_full[1:]  # skip mid (already at end of path1)

            staged_path = np.concatenate([path1, path2], axis=0) if len(path2) > 0 else path1

            if self._check_joint_path_safe(staged_path):
                logger.info(
                    "return_to_home safe joint fallback (2-stage): "
                    "stage1=%d wp (proximal), stage2=%d wp (wrist), total_delta=%.1f°",
                    len(path1),
                    len(path2_full),
                    np.rad2deg(delta),
                )
                return self._execute_waypoints(staged_path, dt)

        logger.warning(
            "Safe joint fallback: path has collisions (self/env), delta=%.1f°",
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

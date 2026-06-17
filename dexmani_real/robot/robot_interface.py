"""RobotInterface — arm + hand 统一上层接口。

控制器和部署模块只通过 RobotInterface 操作硬件，不直接调 XArm7/XHand。
"""

from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.planner.kinematics import XArm7Kinematics
from dexmani_real.planner.planner_types import Pose
from dexmani_real.planner.pose_utils import compose_pose, compute_pose_error, quat_wxyz_to_rot6d
from dexmani_real.planner.workspace_safety import WorkspaceSafety
from dexmani_real.robot.xarm7 import XArm7, XArm7Config
from dexmani_real.robot.xhand import XHand, XHandConfig

if TYPE_CHECKING:
    from dexmani_real.planner.arm_planner import XArm7MotionPlanner



def _dense_interpolate(path: np.ndarray, max_step_rad: float = np.deg2rad(1.0)) -> np.ndarray:
    """将稀疏关节路径插值为稠密路径（每步 ≤ max_step_rad）。"""
    if len(path) <= 1:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step_rad))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


class HandKinematics:
    """手部 FK 辅助：计算指尖在世界坐标系中的位置。

    链式 FK 路径:
      arm_base → [arm_qpos] → EEF → [T_eef_handbase] → hand_base
        → [hand_qpos + hand URDF] → fingertip_i
    """

    def __init__(
        self,
        hand_urdf_path: str,
        fingertip_link_names: list[str] | None = None,
    ) -> None:
        self._model = None
        self._data = None
        self._fingertip_link_ids: list[int] = []
        self._fingertip_link_names: list[str] = []
        self._ready = False

        try:
            import pinocchio
        except ImportError:
            return

        try:
            self._model = pinocchio.buildModelFromUrdf(hand_urdf_path)
            self._data = self._model.createData()
        except Exception:
            return

        if fingertip_link_names is None:
            fingertip_link_names = [
                "right_hand_thumb_rota_tip",
                "right_hand_index_rota_tip",
                "right_hand_mid_tip",
                "right_hand_ring_tip",
                "right_hand_pinky_tip",
            ]

        all_links = list(self._model.names) if hasattr(self._model, "names") else []
        try:
            from pinocchio import FrameType

            for i, frame in enumerate(self._model.frames):
                if frame.type == FrameType.BODY:
                    all_links.append(frame.name)
        except Exception:
            pass

        for name in fingertip_link_names:
            try:
                idx = self._model.getJointId(name)
                if idx < len(self._model.names):
                    self._fingertip_link_ids.append(idx)
                    self._fingertip_link_names.append(name)
            except Exception:
                pass

        self._ready = len(self._fingertip_link_ids) >= 5

    def is_ready(self) -> bool:
        return self._ready

    def compute_tip_positions_in_handbase(
        self, hand_qpos: np.ndarray
    ) -> np.ndarray:
        """返回 (5, 3) 指尖在 hand_base 坐标系中的位置。"""
        if not self._ready:
            return np.full((5, 3), np.nan, dtype=np.float64)

        try:
            import pinocchio
        except ImportError:
            return np.full((5, 3), np.nan, dtype=np.float64)

        q = np.asarray(hand_qpos, dtype=np.float64).reshape(12)
        pinocchio.forwardKinematics(self._model, self._data, q)
        pinocchio.updateFramePlacements(self._model, self._data)

        tips = np.zeros((5, 3), dtype=np.float64)
        for i, link_id in enumerate(self._fingertip_link_ids[:5]):
            placement = self._data.oMi[link_id]
            tips[i] = placement.translation.copy()
        return tips



@dataclass
class RobotState:
    """完整机器人状态 — 来自 RobotInterface.get_state()。

    所有物理量单位标注在注释中。
    """

    # ── Arm 关节 ──
    arm_qpos: np.ndarray          # (7,)  float64  rad
    arm_qvel: np.ndarray          # (7,)  float64  rad/s
    arm_tau: np.ndarray           # (7,)  float64  N·m (实为电机电流)

    # ── EEF 位姿（双表示）──
    eef_pos: np.ndarray           # (3,)  float64  m
    eef_quat_wxyz: np.ndarray     # (4,)  float64
    eef_rot6d: np.ndarray         # (6,)  float64

    # ── Hand 关节 ──
    hand_qpos: np.ndarray         # (12,) float64  rad
    hand_current: np.ndarray      # (12,) float64  mA

    # ── 触觉 ──
    hand_tactile_sum: np.ndarray  # (5,3) float64  N
    hand_tactile_raw: np.ndarray  # (5,120,3) float64

    hand_temperature: np.ndarray  # (12,) float64  °C

    # ── 派生（链式 FK）──
    fingertip_pos: np.ndarray     # (5,3) float64  m (world frame)

    # ── 状态 ──
    arm_connected: bool
    hand_connected: bool
    hand_error: bool
    timestamp: float              # seconds

    def __post_init__(self):
        for field_name, expected_shape in [
            ("arm_qpos", (7,)),
            ("arm_qvel", (7,)),
            ("arm_tau", (7,)),
            ("eef_pos", (3,)),
            ("eef_quat_wxyz", (4,)),
            ("eef_rot6d", (6,)),
            ("hand_qpos", (12,)),
            ("hand_current", (12,)),
            ("hand_tactile_sum", (5, 3)),
            ("hand_tactile_raw", (5, 120, 3)),
            ("hand_temperature", (12,)),
            ("fingertip_pos", (5, 3)),
        ]:
            val = getattr(self, field_name)
            if val is not None:
                arr = np.asarray(val)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"RobotState.{field_name} shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )


@dataclass
class RobotAction:
    """发送给硬件的动作命令。

    arm_qpos_cmd / hand_qpos_cmd: 经过 joint limit + delta limit 后的最终命令。
    target_eef_pos / target_eef_rot6d: IK 前的 EEF 目标（可选）。
    """

    arm_qpos_cmd: np.ndarray             # (7,)  float64  rad
    hand_qpos_cmd: np.ndarray            # (12,) float64  rad

    target_eef_pos: np.ndarray | None = None    # (3,)  float64  m
    target_eef_rot6d: np.ndarray | None = None  # (6,)  float64

    def __post_init__(self):
        for field_name, expected_shape in [
            ("arm_qpos_cmd", (7,)),
            ("hand_qpos_cmd", (12,)),
        ]:
            val = getattr(self, field_name)
            if val is not None:
                arr = np.asarray(val)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"RobotAction.{field_name} shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )



@dataclass
class RobotInterfaceConfig:
    arm: XArm7Config = field(default_factory=XArm7Config)
    hand: XHandConfig = field(default_factory=XHandConfig)

    # Workspace safety
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.0, 0.75],  # x [min, max] m
            [-0.5, 0.5],  # y [min, max] m
            [0.0, 0.6],   # z [min, max] m
        ], dtype=np.float64)
    )

    # Environment collision (table at z=0.0 m in world frame)
    add_table_collision: bool = True
    table_z_world: float = 0.0      # table surface height (world frame, meters)
    table_margin_xy: float = 0.15   # extra margin beyond workspace bounds
    table_layers: int = 5           # number of z-layers for solid volume
    table_layer_spacing: float = 0.01  # spacing between z-layers (meters)
    table_xy_resolution: float = 0.02  # point spacing on each layer (meters)
    table_x_min_clearance: float = 0.15  # minimum x distance from origin (protect robot base)

    # Hand FK
    hand_urdf_path: str = ""
    fingertip_link_names: list[str] = field(default_factory=list)

    # Static transform from EEF to hand base
    T_eef_handbase_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    T_eef_handbase_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )


class RobotInterface:
    """Arm + Hand 统一接口。

    - 控制器和部署模块只通过此类操作硬件
    - hand 断连时降级运行（arm 仍可工作）
    - send_action() 返回 dict[str, bool] 区分子设备状态
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
        self.workspace = WorkspaceSafety(config.workspace_bounds)

        self.arm = XArm7(config.arm)
        self.hand = XHand(config.hand)

        # 验证 home EEF 在 workspace 内（提前发现配置错误）
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

        # 设置桌面碰撞几何（plan_screw/plan_qpos 会自动避开）
        if config.add_table_collision and self.planner is not None:
            self._setup_table_collision(
                table_z=config.table_z_world,
                margin_xy=config.table_margin_xy,
                n_layers=config.table_layers,
                layer_spacing=config.table_layer_spacing,
                xy_resolution=config.table_xy_resolution,
                x_min_clearance=config.table_x_min_clearance,
            )

        # 手部运动学
        self.hand_kinematics: HandKinematics | None = None
        if config.hand_urdf_path:
            hk = HandKinematics(config.hand_urdf_path, config.fingertip_link_names or None)
            if hk.is_ready():
                self.hand_kinematics = hk

    # ------------------------------------------------------------
    # 生命周期

    def connect(self) -> dict[str, bool]:
        """连接 arm + hand。返回 {"arm": bool, "hand": bool}。"""
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
        """读取 arm + hand 状态，含 FK 计算。"""
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
            eef_pos = np.full(3, np.nan, dtype=np.float64)
            eef_quat_wxyz = np.full(4, np.nan, dtype=np.float64)
            eef_rot6d = np.full(6, np.nan, dtype=np.float64)

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
            hand_state.get("temperature", np.full(12, np.nan)),
            dtype=np.float64,
        )

        # 指尖世界坐标
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
        """发送 arm + hand 动作。

        Returns:
            {"arm_ok": bool, "hand_ok": bool,
             "arm_cmd": ndarray | None,   # (7,) post-clip 实际发送值
             "hand_cmd": ndarray | None}  # (12,) post-clip 实际发送值

        arm_cmd/hand_cmd 是经过 joint limit + delta limit 裁剪后的实际命令值。
        发送失败时为 None。录制时应使用这些 post-clip 值而非 IK 原始输出。
        """
        result: dict = {}

        # FK 验证实际关节命令的 EEF 位姿是否在 workspace 内
        if np.all(np.isfinite(action.arm_qpos_cmd)):
            cmd_eef_pose = self.kinematics.compute_eef_pose_world(action.arm_qpos_cmd)
            if not self.workspace.check(cmd_eef_pose.p):
                warnings.warn(
                    f"Command EEF {np.round(cmd_eef_pose.p, 4)} m "
                    f"outside workspace bounds, send_action proceeds "
                    f"(enforcement at controller level)"
                )

        arm_ok = self.arm.send_action(action.arm_qpos_cmd)
        hand_ok = self.hand.send_action(action.hand_qpos_cmd)

        result["arm_ok"] = arm_ok
        result["hand_ok"] = hand_ok
        result["arm_cmd"] = self.arm.last_qpos_cmd.copy() if arm_ok else None
        result["hand_cmd"] = self.hand.last_qpos_cmd.copy() if hand_ok else None
        return result


    def reset_hand(self) -> bool:
        """复位手部到 home 位置。hand 断连时返回 False。"""
        if not self.hand.is_connected():
            return False
        return self.hand.reset()

    def return_to_home(
        self,
        use_planning: bool = True,
        cancel_event: Any = None,
    ) -> bool:
        """两阶段 return_home：EEF 归位 → 冗余关节归位。

        Phase 1: plan_path(home_eef) — Cartesian 路径把 EEF 移回 home。
        Phase 2: 关节空间插值 — 当前 qpos → home_qpos，逐点碰撞检测。
        use_planning=False 时走 direct reset（直线关节空间 + hand reset）。

        SIGINT (Ctrl+C) 会设置 cancel_event 来中止路径执行。
        """
        # Install SIGINT handler so Ctrl+C cancels waypoint execution
        import signal as _signal

        def _on_sigint(signum, frame):
            if cancel_event is not None:
                cancel_event.set()

        old_handler = _signal.signal(_signal.SIGINT, _on_sigint)
        try:
            # 1. Arm not connected → bail out
            if not self.arm.is_connected():
                return False

            # 2. Read current qpos; NaN → fallback
            arm_state = self.arm.get_state()
            current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
            if not np.all(np.isfinite(current_qpos)):
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

            # 5. FK(home_qpos) → home EEF pose, workspace check/clamp
            home_qpos = self.arm.config.init_qpos.copy()
            home_eef_pose = self.kinematics.compute_eef_pose_world(home_qpos)
            if not self.workspace.check(home_eef_pose.p):
                warnings.warn(
                    f"Home EEF position {np.round(home_eef_pose.p, 4)} "
                    "is outside workspace, clamping"
                )
                home_eef_pose.p = self.workspace.clamp(home_eef_pose.p)

            # 6. Already at home? (joint error < 1 deg, covers redundant IK)
            if float(np.max(np.abs(current_qpos - home_qpos))) < np.deg2rad(1.0):
                hand_ok = self.hand.reset() if self.hand.is_connected() else True
                return hand_ok

            # ── Phase 1: EEF 路径 → home EEF ──
            try:
                result = self.planner.plan_path(home_eef_pose, current_qpos)
            except Exception:
                traceback.print_exc()
                return self._return_to_home_direct()

            if not result.success or result.qpos_path is None or len(result.qpos_path) == 0:
                return self._return_to_home_direct()

            dt = float(self.arm.config.dt)

            # Phase 1 前先把手部复位到 home。
            # 规划器 URDF 模型中手部是固定默认构型，真机手部可能被遥操作
            # 驱动到任意构型（手指张开/握拳等），二者不一致会导致碰撞检测
            # 失效。这里先复位手部使真机构型与规划模型一致。
            if self.hand.is_connected():
                self.hand.reset()
                time.sleep(dt * 10)  # 等手部收敛

            # 复位后验证指尖确实在桌面之上（安全冗余）
            desk_z = self.config.table_z_world if self.config.add_table_collision else 0.0
            if self.planner is not None and self.config.add_table_collision:
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
                        warnings.warn(
                            f"Fingertips still below desk after hand reset "
                            f"(first_z={min_z_first:.3f}m last_z={min_z_last:.3f}m)"
                        )

            phase1_completed = True
            # 插值为 1° 步长稠密路径，避免 _limit_joint_step 裁剪大跳变
            dense_path = _dense_interpolate(result.qpos_path)
            for waypoint in dense_path:
                if (cancel_event is not None and cancel_event.is_set()) or self.arm.is_error():
                    phase1_completed = False
                    break
                if not self.arm.send_action(waypoint):
                    phase1_completed = False
                    break
                time.sleep(dt)

            # 闭环等待 servo 收敛到路径终点
            if phase1_completed:
                target_qpos = result.qpos_path[-1]
                max_wait = max(dt * 5, float(np.max(np.abs(
                    target_qpos - result.qpos_path[0]))) / float(np.min(
                    self.arm.config.max_qvel)) * 5.0)
                poll_interval = dt * 2
                elapsed = 0.0
                converged = False
                while elapsed < max_wait:
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                    try:
                        poll_qpos = np.asarray(
                            self.arm.get_state()["qpos"], dtype=np.float64)
                        if not np.all(np.isfinite(poll_qpos)):
                            continue
                        err = float(np.max(np.abs(poll_qpos - target_qpos)))
                        if err < np.deg2rad(3.0):
                            converged = True
                            break
                    except Exception:
                        continue
                if not converged:
                    warnings.warn(
                        f"Phase 1 convergence timeout after {max_wait:.1f}s, "
                        f"skipping Phase 2 joint fine-tuning"
                    )
                    phase1_completed = False

            # ── Phase 2: 关节空间归位 → home_qpos ──
            if phase1_completed:
                arm_state = self.arm.get_state()
                current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
                if np.all(np.isfinite(current_qpos)):
                    joint_delta = float(np.max(np.abs(current_qpos - home_qpos)))
                    if joint_delta > np.deg2rad(0.5):
                        joint_path = self._safe_joint_path(current_qpos, home_qpos)
                        if joint_path is not None:
                            for waypoint in joint_path:
                                if (cancel_event is not None and cancel_event.is_set()) or self.arm.is_error():
                                    break
                                if not self.arm.send_action(waypoint):
                                    break
                                time.sleep(dt)
                        else:
                            # Joint path would self-collide → skip Phase 2.
                            # _return_to_home_direct() 走的是同一条关节直线路径
                            # （只是由 SDK 轨迹生成器执行），并不会更安全。
                            # Phase 1 已把 EEF 归位，跳过 Phase 2 最多损失
                            # 冗余关节的精确对齐，不会造成碰撞风险。
                            warnings.warn(
                                "Joint-space home path self-collides, "
                                "skipping Phase 2 (EEF already at home from Phase 1)"
                            )

            # Hand reset (degraded if hand not connected)
            hand_ok = self.hand.reset() if self.hand.is_connected() else True
            arm_ok = not self.arm.is_error()
            return arm_ok and hand_ok
        finally:
            _signal.signal(_signal.SIGINT, old_handler)

    def _safe_joint_path(
        self, start: np.ndarray, goal: np.ndarray, max_step_rad: float = np.deg2rad(2.0)
    ) -> np.ndarray | None:
        """线性插值 start → goal，逐点碰撞检测（自碰撞 + 环境碰撞）。
        不安全返回 None。"""
        dist = float(np.max(np.abs(goal - start)))
        n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
        path = np.array([start + (k / (n - 1)) * (goal - start) for k in range(n)])

        if self.planner is None:
            warnings.warn(
                "_safe_joint_path called without planner, cannot check collisions"
            )
            return None

        profile = self.planner.planning_profile
        if profile.check_self_collision:
            if any(self.planner.has_self_collision(q) for q in path):
                return None
        if profile.check_env_collision:
            if any(self.planner.has_env_collision(q) for q in path):
                return None
        return path

    def _return_to_home_direct(self) -> bool:
        """Fallback: direct arm.reset() + hand reset (straight-line in joint space)."""
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
        print(
            f"[RobotInterface] Table collision: {points.shape[0]} points, "
            f"{n_layers} layers, z=[{zs[-1]:.3f}, {zs[0]:.3f}] m, "
            f"xy=[{x_min:.2f},{x_max:.2f}]x[{y_min:.2f},{y_max:.2f}] m"
        )

    def _check_fingertips_above_desk(
        self, arm_qpos: np.ndarray, hand_qpos: np.ndarray, desk_z: float = 0.0,
    ) -> tuple[bool, float]:
        """用实际 hand_qpos + arm waypoint FK 检查指尖是否在桌面之上。

        Returns: (all_above, min_z).  仅用于 hand_kinematics 可用时的执行层校验。
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
            return np.full((5, 3), np.nan, dtype=np.float64)

        if not np.all(np.isfinite(eef_pos)) or not np.all(np.isfinite(hand_qpos)):
            return np.full((5, 3), np.nan, dtype=np.float64)

        tips_in_handbase = self.hand_kinematics.compute_tip_positions_in_handbase(hand_qpos)
        if not np.all(np.isfinite(tips_in_handbase)):
            return np.full((5, 3), np.nan, dtype=np.float64)

        T_world_eef = Pose(p=eef_pos, q=eef_quat_wxyz)

        T_eef_handbase = Pose(
            p=self.config.T_eef_handbase_pos,
            q=self.config.T_eef_handbase_quat_wxyz,
        )

        # 每个指尖
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

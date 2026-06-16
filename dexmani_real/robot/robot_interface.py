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
            [0.2, 0.7],   # x [min, max] m
            [-0.3, 0.3],  # y [min, max] m
            [0.0, 0.6],   # z [min, max] m
        ], dtype=np.float64)
    )

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
        """
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

        # 6. Already at home? (< 5 mm pos error, < 0.05 rad rot error)
        current_eef = self.kinematics.compute_eef_pose_world(current_qpos)
        pos_err, rot_err = compute_pose_error(home_eef_pose, current_eef)
        if pos_err <= 0.005 and rot_err <= 0.05:
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
        phase1_completed = True
        for waypoint in result.qpos_path:
            if (cancel_event is not None and cancel_event.is_set()) or self.arm.is_error():
                phase1_completed = False
                break
            self.arm.send_action(waypoint)
            time.sleep(dt)

        # 等 servo 收敛到最后一个 waypoint，再读当前 qpos 做 Phase 2
        if phase1_completed:
            time.sleep(dt * 3)

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
                            self.arm.send_action(waypoint)
                            time.sleep(dt)

        # Hand reset (degraded if hand not connected)
        hand_ok = self.hand.reset() if self.hand.is_connected() else True
        arm_ok = not self.arm.is_error()
        return arm_ok and hand_ok

    def _safe_joint_path(
        self, start: np.ndarray, goal: np.ndarray, max_step_rad: float = np.deg2rad(2.0)
    ) -> np.ndarray | None:
        """线性插值 start → goal，逐点碰撞检测。不安全返回 None。"""
        dist = float(np.max(np.abs(goal - start)))
        n = max(2, int(np.ceil(dist / max_step_rad)) + 1)
        path = np.array([start + (k / (n - 1)) * (goal - start) for k in range(n)])

        if self.planner is not None and self.planner.planning_profile.check_self_collision:
            if any(self.planner.has_self_collision(q) for q in path):
                return None
        return path

    def _return_to_home_direct(self) -> bool:
        """Fallback: direct arm.reset() + hand reset (straight-line in joint space)."""
        arm_ok = self.arm.reset()
        hand_ok = self.hand.reset() if self.hand.is_connected() else True
        return arm_ok and hand_ok

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

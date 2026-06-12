"""RobotInterface — arm + hand 统一上层接口。

控制器和部署模块只通过 RobotInterface 操作硬件，不直接调 XArm7/XHand。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.robot.planner.kinematics import XArm7Kinematics
from dexmani_real.robot.planner.planner_types import Pose
from dexmani_real.robot.planner.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.robot.planner.workspace_safety import WorkspaceSafety
from dexmani_real.robot.xarm7 import XArm7, XArm7Config
from dexmani_real.robot.xhand import XHand, XHandConfig


# ---------------------------------------------------------------------------
# 手部运动学辅助类
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 统一状态与动作类型
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# RobotInterface 复合接口
# ---------------------------------------------------------------------------


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
    ) -> None:
        self.config = config
        self.kinematics = kinematics
        self.workspace = WorkspaceSafety(config.workspace_bounds)

        self.arm = XArm7(config.arm)
        self.hand = XHand(config.hand)

        # 手部运动学
        self.hand_kinematics: HandKinematics | None = None
        if config.hand_urdf_path:
            hk = HandKinematics(config.hand_urdf_path, config.fingertip_link_names or None)
            if hk.is_ready():
                self.hand_kinematics = hk

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

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

    def is_error(self) -> bool:
        return self.arm.is_error() or self.hand.is_error()

    def clear_error(self) -> bool:
        arm_ok = self.arm.clear_error()
        hand_ok = self.hand.clear_error()
        return arm_ok and hand_ok

    def emergency_stop(self) -> None:
        """Arm + Hand 同时急停。"""
        self.arm.stop()
        self.hand.stop()

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 动作发送
    # ------------------------------------------------------------------

    def send_action(self, action: RobotAction) -> dict[str, bool]:
        """发送 arm + hand 动作。返回 {"arm": bool, "hand": bool}。

        EEF 目标必须在 workspace 内。
        """
        result: dict[str, bool] = {}

        # Workspace 检查
        target_pos = action.target_eef_pos
        if target_pos is not None and not self.workspace.check(target_pos):
            target_pos = self.workspace.clamp(target_pos)

        result["arm"] = self.arm.send_action(action.arm_qpos_cmd)
        result["hand"] = self.hand.send_action(action.hand_qpos_cmd)
        return result

    # ------------------------------------------------------------------
    # 复位
    # ------------------------------------------------------------------

    def return_to_home(
        self,
        use_planning: bool = True,
        cancel_event: Any = None,
    ) -> bool:
        """路径规划回 home + hand 复位。

        use_planning=True: 使用 planner.plan_path() 规划路径
        规划失败时 fallback 直线 reset()
        """
        arm_ok = self.arm.reset()
        hand_ok = self.reset_hand()
        return arm_ok and hand_ok

    def reset_hand(self) -> bool:
        return self.hand.reset()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_fingertip_pos(
        self,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
        hand_qpos: np.ndarray,
    ) -> np.ndarray:
        """计算 5 个指尖在世界坐标系中的位置。

        链式 FK: T_world_fingertip = T_world_eef @ T_eef_handbase @ T_handbase_fingertip
        """
        if self.hand_kinematics is None or not self.hand_kinematics.is_ready():
            return np.full((5, 3), np.nan, dtype=np.float64)

        if not np.all(np.isfinite(eef_pos)) or not np.all(np.isfinite(hand_qpos)):
            return np.full((5, 3), np.nan, dtype=np.float64)

        tips_in_handbase = self.hand_kinematics.compute_tip_positions_in_handbase(hand_qpos)
        if not np.all(np.isfinite(tips_in_handbase)):
            return np.full((5, 3), np.nan, dtype=np.float64)

        # T_world_eef
        from dexmani_real.robot.planner.pose_utils import compose_pose

        T_world_eef = Pose(p=eef_pos.copy(), q=eef_quat_wxyz.copy())

        # T_eef_handbase (static)
        T_eef_handbase = Pose(
            p=self.config.T_eef_handbase_pos.copy(),
            q=self.config.T_eef_handbase_quat_wxyz.copy(),
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

"""XArm7_XHand 仿真 → RobotInterface 接口适配器。

将 SAPIEN 仿真类包装为与真机驱动一致的接口，使 controller 和 test
可以在仿真/真机之间无缝切换。

无硬件依赖: SAPIEN headless 模式可在 CI 中运行 (无 GPU/显示器)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.robot.model.constructor import add_base_components, setup_scene
from dexmani_real.robot.model.xarm7_xhand import XArm7_XHand


@dataclass
class SimRobotConfig:
    dt: float = 1.0 / 50.0
    time_step: float = 1.0 / 240.0      # SAPIEN 物理步长 (通常 > 控制频率)
    headless: bool = True
    arm_home_qpos: np.ndarray = field(
        default_factory=lambda: np.array(
            [-np.pi / 6, -np.pi / 4, 0, np.deg2rad(20), -np.pi, np.deg2rad(25), 0]
        )
    )


class SimRobotInterface:
    """SAPIEN 仿真 → 真机 RobotInterface 接口适配。

    提供与 RobotInterface (CLAUDE.md Section 2.8) 一致的接口:
      connect() → bool
      get_state() → dict (含 arm_qpos, hand_qpos, eef_pos 等)
      send_action(action: np.ndarray) → bool
      reset() → bool
      stop() / is_connected() / is_error() / clear_error()
    """

    def __init__(self, config: SimRobotConfig | None = None):
        self.config = config or SimRobotConfig()
        self.scene = None
        self.robot: XArm7_XHand | None = None
        self.step_count = 0

        self.connected_flag = False
        self.error_state = False
        self.last_error_message = ""
        self.last_action_code: int | None = None
        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self.scene = setup_scene(time_step=self.config.time_step)
            if not self.config.headless:
                add_base_components(self.scene)
            self.robot = XArm7_XHand(
                self.scene,
                disable_self_collision=True,
                arm_home_qpos=self.config.arm_home_qpos.copy(),
            )
        except Exception as e:
            self.error_state = True
            self.last_error_message = f"sim setup failed: {e}"
            return False

        self.last_qpos_cmd = self.robot.get_qpos().copy()
        self.last_cmd_time = time.time()
        self.connected_flag = True
        self.error_state = False
        return True

    def disconnect(self) -> None:
        self.connected_flag = False
        self.scene = None
        self.robot = None

    def is_connected(self) -> bool:
        return self.robot is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.robot is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        return False

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_message = ""
        return self.is_connected()

    def stop(self) -> bool:
        """软停 — 发送当前位置作为目标（原地保持）。"""
        if self.robot is None:
            return False
        qpos = self.robot.get_qpos()
        self.robot.apply_action(qpos)
        return True

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------

    def get_state(self, full: bool = False) -> dict[str, Any]:
        if self.robot is None:
            return {
                "qpos": np.full(19, np.nan),
                "qvel": np.full(19, np.nan),
                "timestamp": time.time(),
            }

        qpos = self.robot.get_qpos()          # (19,) [arm7 + hand12]
        eef_pose = self.robot.get_eef_pose()  # sapien.Pose

        state: dict[str, Any] = {
            "arm_qpos": qpos[:7].copy(),       # rad
            "hand_qpos": qpos[7:].copy(),      # rad
            "eef_pos": np.array(eef_pose.p, dtype=np.float64),   # m
            "eef_quat_wxyz": np.array(eef_pose.q, dtype=np.float64),  # w,x,y,z
            "qvel": np.zeros(19, dtype=np.float64),
            "timestamp": time.time(),
        }

        if full:
            state.update({
                "qpos_full": qpos.copy(),
                "qlimits": self.robot.qlimits.copy() if self.robot.qlimits is not None else None,
                "connected_flag": self.connected_flag,
                "error_state": self.error_state,
                "last_error_message": self.last_error_message,
                "step_count": self.step_count,
            })
        return state

    # ------------------------------------------------------------------
    # 动作发送
    # ------------------------------------------------------------------

    def send_action(self, action: np.ndarray) -> bool:
        if self.robot is None:
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(19)

        # joint limit clip (使用仿真 URDF 限位)
        if self.robot.qlimits is not None:
            qmin = self.robot.qlimits[:, 0]
            qmax = self.robot.qlimits[:, 1]
            clipped = np.clip(target_qpos, qmin, qmax)
            self.last_joint_limit_clipped = not np.allclose(target_qpos, clipped)
            target_qpos = clipped

        self.robot.apply_action(target_qpos)
        self._step_physics()

        self.last_qpos_cmd = target_qpos.copy()
        self.last_cmd_time = time.time()
        self.last_action_code = 0
        self.step_count += 1
        return True

    # ------------------------------------------------------------------
    # 复位
    # ------------------------------------------------------------------

    def reset(self, target: np.ndarray | None = None) -> bool:
        if self.robot is None:
            return False

        if target is not None:
            self.robot.set_qpos(target)
            self.robot.balance_passive_force()
            self.robot.apply_action(target)
        else:
            self.robot.reset(random_init=False)

        self._step_physics()
        self.last_qpos_cmd = self.robot.get_qpos().copy()
        self.last_cmd_time = time.time()
        return True

    def return_to_home(self) -> bool:
        return self.reset()

    def emergency_stop(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _step_physics(self, n: int = 5):
        """推进物理仿真。n=5: 240Hz → 48Hz 有效控制频率。"""
        for _ in range(n):
            self.scene.step()

    def get_full_qpos(self) -> np.ndarray:
        """返回完整 19-DOF qpos [arm7, hand12]。"""
        if self.robot is None:
            return np.full(19, np.nan)
        return self.robot.get_qpos().copy()

    # ------------------------------------------------------------------
    # 仿真专属验证方法
    # ------------------------------------------------------------------

    def validate_fk_consistency(self) -> dict[str, Any]:
        """验证 FK 一致性: link poses vs forward_kinematics。"""
        if self.robot is None:
            return {"ok": False, "error": "not connected"}

        qpos = self.robot.get_qpos()
        link_poses_real = self.robot.get_link_poses(self.robot.fingertip_link_names)
        link_poses_fk = self.robot.forward_kinematics(qpos, self.robot.fingertip_link_names)

        max_err = np.max(np.abs(link_poses_real - link_poses_fk))
        return {"ok": max_err < 1e-4, "max_error": float(max_err)}

    def validate_ik_roundtrip(self, n_tests: int = 20) -> dict[str, Any]:
        """验证 IK 往返一致性: IK(FK(q)) ≈ q。"""
        if self.robot is None:
            return {"ok": False, "error": "not connected"}

        max_err = 0.0
        qpos = self.robot.get_qpos()
        eef_pose = self.robot.get_eef_pose()

        for i in range(n_tests):
            try:
                ik_qpos = self.robot.inverse_kinematics(eef_pose, full_qpos_init=qpos)
                err = np.max(np.abs(ik_qpos[:7] - qpos[:7]))  # 仅比较 arm 关节
                max_err = max(max_err, err)
            except RuntimeError:
                return {"ok": False, "error": f"IK failed at test {i}"}

        return {"ok": max_err < 0.1, "max_error": float(max_err), "n_tests": n_tests}


def example():
    sim = SimRobotInterface(SimRobotConfig(headless=True))
    if not sim.connect():
        print(f"connect failed: {sim.last_error_message}")
        return

    try:
        print("connected:", sim.is_connected())

        state = sim.get_state(full=True)
        print(f"arm_qpos: {np.round(state['arm_qpos'], 3)}")
        print(f"eef_pos: {np.round(state['eef_pos'], 3)}")

        # FK 一致性
        fk = sim.validate_fk_consistency()
        print(f"FK consistency: ok={fk['ok']}, max_err={fk['max_error']:.6f}")

        # IK 往返
        ik = sim.validate_ik_roundtrip()
        print(f"IK roundtrip: ok={ik['ok']}, max_err={ik.get('max_error', 0):.6f}")

        # stay-in-place
        ok = sim.send_action(sim.get_full_qpos())
        print(f"send_action: {ok}")

        # small-move
        target = sim.get_full_qpos() + np.deg2rad(2.0)
        ok = sim.send_action(target)
        print(f"small-move: {ok}")

        # reset
        ok = sim.reset()
        print(f"reset: {ok}")
        print(f"  arm_qpos after: {np.round(sim.get_state()['arm_qpos'], 3)}")

        print("\nall sim tests passed")

    finally:
        sim.disconnect()


if __name__ == "__main__":
    example()

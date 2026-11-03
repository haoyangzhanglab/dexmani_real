"""限制速度、加速度与 jerk；换目标或制动时沿用上一参考状态。"""

import numpy as np
from ruckig import (
    ControlInterface,
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
    Synchronization,
)

from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.robot.commands import validate_arm_target


# 仅容纳轨迹计算的浮点舍入误差（rad），不是运动或到位容差。
_POSITION_ROUNDOFF_RAD = 1e-12


class ArmTrajectory:
    def __init__(self, cfg: ArmConfig, q: np.ndarray, v: np.ndarray) -> None:
        self.lower = np.asarray(cfg.joint_limit_lower)
        self.upper = np.asarray(cfg.joint_limit_upper)
        self.otg = Ruckig(7, 1 / cfg.servo_hz)
        self.input = InputParameter(7)
        self.output = OutputParameter(7)
        self.input.max_velocity = [cfg.max_joint_velocity_rad_per_s] * 7
        self.input.max_acceleration = [cfg.max_joint_acceleration_rad_per_s2] * 7
        self.input.max_jerk = [cfg.max_joint_jerk_rad_per_s3] * 7
        self.input.target_velocity = [0.0] * 7
        self.input.current_position = validate_arm_target(q, self.lower, self.upper).tolist()
        if np.any(np.abs(v) > cfg.max_joint_velocity_rad_per_s + 1e-8):
            raise ValueError("initial arm velocity exceeds limit")
        # SDK 停稳反馈仍有浮点残差；仅初始化时清理 <1e-12 rad/s 的噪声，
        # 避免同位目标触发 Ruckig 的近零时长同步退化。
        self.input.current_velocity = np.where(np.abs(v) < 1e-12, 0.0, v).tolist()
        self.input.target_position = self.input.current_position

    def step(self, goal: np.ndarray | None = None, *, hold: bool = False) -> np.ndarray:
        """计算候选；input.current_* 始终保留最近一次成功发送的参考。"""
        # 保持请求只制动；各关节独立停稳，不继续追赶此前积累的位置目标。
        if hold:
            self.input.control_interface = ControlInterface.Velocity
            self.input.synchronization = Synchronization.No
        elif goal is not None:
            self.input.control_interface = ControlInterface.Position
            self.input.synchronization = Synchronization.Time
            self.input.target_position = goal.tolist()
        if self.otg.update(self.input, self.output) not in (Result.Working, Result.Finished):
            raise ValueError("Ruckig calculation failed")
        if self.output.new_calculation:
            # 段端点及 v=0 处覆盖完整制动距离；库的 position_extrema 会漏掉恒加速度段内极值。
            for i, profile in enumerate(self.output.trajectory.profiles[0]):
                extrema = [self.input.current_position[i], *profile.p]
                for segment in (profile.brake, profile):
                    for p, v, a, j, t in zip(segment.p, segment.v, segment.a, segment.j, segment.t):
                        if t == 0:
                            continue
                        extrema.append(p)
                        roots: tuple[float, ...]
                        if j == 0:
                            roots = (-v / a,) if a != 0 else ()
                        else:
                            d = a * a - 2 * j * v
                            roots = ((-a - np.sqrt(d)) / j, (-a + np.sqrt(d)) / j) if d >= 0 else ()
                        for tau in roots:
                            if 0 < tau < t:
                                extrema.append(p + tau * (v + tau * (a / 2 + tau * j / 6)))
                if (
                    not np.isfinite(extrema).all()
                    or min(extrema) < self.lower[i] - _POSITION_ROUNDOFF_RAD
                    or max(extrema) > self.upper[i] + _POSITION_ROUNDOFF_RAD
                ):
                    raise ValueError(
                        f"reference braking/position extremum outside joint {i + 1} limits"
                    )
        q = np.asarray(self.output.new_position)
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - _POSITION_ROUNDOFF_RAD)
            or np.any(q > self.upper + _POSITION_ROUNDOFF_RAD)
        ):
            raise ValueError("reference position outside joint limits")
        # 完整轨迹已通过极值检查；只归一采样点的舍入越界。
        return np.clip(q, self.lower, self.upper)

    def commit(self, q: np.ndarray) -> None:
        """SDK 发送成功后，提交 step 返回的位置及对应速度、加速度。"""
        # update 只推进 output 和库内比较缓存。未 commit 时，下次 update
        # 会因 input 与缓存不同而从已提交参考重算，无需回滚或重置轨迹。
        self.output.pass_to_input(self.input)
        self.input.current_position = q.tolist()

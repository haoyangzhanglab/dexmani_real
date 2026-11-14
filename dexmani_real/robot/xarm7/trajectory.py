"""限制速度、加速度与 jerk；换目标或制动时沿用上一参考状态。"""

import numpy as np
from ruckig import (
    ControlInterface,
    DurationDiscretization,
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
    Synchronization,
    Trajectory,
)

from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.robot.commands import arm_reference_from_feedback

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
        self.input.current_position = arm_reference_from_feedback(q, self.lower, self.upper).tolist()
        if np.any(np.abs(v) > cfg.max_joint_velocity_rad_per_s + 1e-8):
            raise ValueError("initial arm velocity exceeds limit")
        # SDK 停稳反馈仍有浮点残差；仅初始化时清理 <1e-12 rad/s 的噪声，
        # 避免同位目标触发 Ruckig 的近零时长同步退化。
        self.input.current_velocity = np.where(np.abs(v) < 1e-12, 0.0, v).tolist()
        self.input.target_position = self.input.current_position

    def step(self, goal: np.ndarray | None = None, *, hold: bool = False) -> np.ndarray:
        """计算并推进参考；未发送或发送失败后必须丢弃本轨迹。"""
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
            validate_extrema(self.output.trajectory, self.input.current_position, self.lower, self.upper)
        q = np.asarray(self.output.new_position)
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - _POSITION_ROUNDOFF_RAD)
            or np.any(q > self.upper + _POSITION_ROUNDOFF_RAD)
        ):
            raise ValueError("reference position outside joint limits")
        # 完整轨迹已通过极值检查；只归一采样点的舍入越界。
        q = np.clip(q, self.lower, self.upper)
        self.output.pass_to_input(self.input)
        self.input.current_position = q.tolist()
        return q


class PositionLimitError(ValueError):
    """完整参考轨迹越过操作位置限位。"""


def validate_extrema(trajectory, start, lower, upper):
    # 段端点及 v=0 处覆盖完整制动距离；库的 position_extrema 会漏掉恒加速度段内极值。
    for i, profile in enumerate(trajectory.profiles[0]):
        extrema = [start[i], *profile.p]
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
        if not np.isfinite(extrema).all():
            raise RuntimeError("non-finite Ruckig profile")
        if (
            min(extrema) < lower[i] - _POSITION_ROUNDOFF_RAD
            or max(extrema) > upper[i] + _POSITION_ROUNDOFF_RAD
        ):
            raise PositionLimitError(
                f"reference braking/position extremum outside joint {i + 1} limits"
            )


def calculate_home_segment(cfg, q, v, target):
    """计算一段零加速度起步、零速度／加速度到达的离线轨迹。"""
    # 配置与反馈由 HOME 入口校验，初速还需满足规划器的 HOME 限速。
    inp = InputParameter(7)
    inp.current_position = np.asarray(q).tolist()
    inp.current_velocity = np.where(np.abs(v) < 1e-12, 0., v).tolist()
    inp.current_acceleration = [0.] * 7
    inp.target_position = np.asarray(target).tolist()
    inp.target_velocity = inp.target_acceleration = [0.] * 7
    inp.max_velocity = [cfg.homing.max_speed_rad_per_s] * 7
    inp.max_acceleration = [cfg.homing.max_acceleration_rad_per_s2] * 7
    inp.max_jerk = [np.deg2rad(cfg.homing.max_jerk_deg_per_s3)] * 7
    inp.duration_discretization = DurationDiscretization.Discrete
    trajectory = Trajectory(7)
    result = Ruckig(7, 1 / cfg.servo_hz).calculate(inp, trajectory)
    # 当前 Ruckig 不检查位置限位，路径越限由 validate_extrema 判定。
    # 求解器错误不能证明路径不可行，不能据此改走 Via。
    if result not in (Result.Working, Result.Finished):
        raise RuntimeError(f"HOME Ruckig failed: {result}")
    return trajectory

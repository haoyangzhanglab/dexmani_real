"""Ruckig 连续参考；运动中替换目标不以实测反馈重置轨迹。"""

import numpy as np

from dexmani_real.robot.commands import validate_arm_target


class ArmTrajectory:
    def __init__(self, cfg, q, v, now_ns):
        from ruckig import InputParameter, Ruckig

        self.lower = np.asarray(cfg.joint_limit_lower)
        self.upper = np.asarray(cfg.joint_limit_upper)
        self.otg = Ruckig(7)
        self.input = InputParameter(7)
        self.input.max_velocity = [cfg.max_joint_velocity_rad_per_s] * 7
        self.input.max_acceleration = [cfg.max_joint_acceleration_rad_per_s2] * 7
        # Ruckig 0.17.3 的默认无限 jerk 选择二阶轨迹。
        self.input.target_velocity = [0.0] * 7
        self.q = validate_arm_target(q, self.lower, self.upper).copy()
        self.v = np.array(v, dtype=np.float64, copy=True)
        if self.v.shape != (7,) or not np.isfinite(self.v).all():
            raise ValueError("invalid initial arm velocity")
        if np.any(np.abs(self.v) > cfg.max_joint_velocity_rad_per_s + 1e-8):
            raise ValueError("initial arm velocity exceeds limit")
        self.origin_ns = self.sample_ns = int(now_ns)
        self._plan(self.q, now_ns)

    def step(self, now_ns, goal=None) -> np.ndarray:
        if now_ns != self.sample_ns:
            q, v, a = (
                np.asarray(value)
                for value in self.trajectory.at_time((now_ns - self.origin_ns) / 1e9)
            )
            if not all(np.isfinite(value).all() for value in (q, v, a)):
                raise ValueError("non-finite arm reference")
            self.q, self.v, self.sample_ns = q, v, int(now_ns)

        if goal is not None and not np.array_equal(goal, self.goal):
            self._plan(goal, now_ns)
        return self.q

    def _plan(self, goal, now_ns):
        from ruckig import Result, Trajectory

        self.input.current_position = self.q.tolist()
        self.input.current_velocity = self.v.tolist()
        self.input.target_position = goal.tolist()
        self.otg.validate_input(self.input, True, True)
        trajectory = Trajectory(7)
        if self.otg.calculate(self.input, trajectory) not in (Result.Working, Result.Finished):
            raise ValueError("Ruckig calculation failed")
        # Ruckig 不强制位置限位，必须检查包含制动段的整条轨迹极值。
        for i, bounds in enumerate(trajectory.position_extrema):
            if (
                not np.isfinite([bounds.min, bounds.max]).all()
                or bounds.min < self.lower[i]
                or bounds.max > self.upper[i]
            ):
                raise ValueError(
                    f"reference braking/position extremum outside joint {i + 1} limits"
                )
        self.trajectory = trajectory
        self.origin_ns = int(now_ns)
        self.goal = goal.copy()

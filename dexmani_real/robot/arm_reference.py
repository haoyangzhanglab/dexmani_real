"""Local seven-joint Ruckig reference, sampled on the host monotonic clock.

Only activated goals enter this object. No measured-state reset, wrapping, clipping,
waypoints, network service, or second filter is used during a streaming interval.
"""

import numpy as np

from dexmani_real.robot.command_validation import check_arm_target


class ArmReference:
    def __init__(self, cfg, q, v, now_ns):
        from ruckig import InputParameter, Ruckig

        self.lower = np.asarray(cfg.joint_limit_lower)
        self.upper = np.asarray(cfg.joint_limit_upper)
        self.vmax = cfg.max_joint_velocity_rad_per_s
        self.amax = cfg.max_joint_acceleration_rad_per_s2
        self.otg = Ruckig(7)
        self.input = InputParameter(7)
        self.input.max_velocity = [self.vmax] * 7
        self.input.max_acceleration = [self.amax] * 7
        # Community 0.17.3: omitted max_jerk = infinity selects second-order OTG.
        self.input.target_velocity = [0.0] * 7
        self.goal = None
        self.q, self.v = self._state(q, v, np.zeros(7))[:2]
        self.origin_ns = self.sample_ns = int(now_ns)
        self.replace(q, now_ns)

    def _state(self, q, v, a):
        q, v, a = (np.asarray(x, dtype=np.float64) for x in (q, v, a))
        issue = check_arm_target(q, joint_limit_lower_rad=self.lower,
                                 joint_limit_upper_rad=self.upper)
        if issue or any(x.shape != (7,) or not np.isfinite(x).all() for x in (v, a)):
            raise ValueError(f"invalid arm reference: {issue or 'velocity/acceleration'}")
        if np.any(np.abs(v) > self.vmax + 1e-8) or np.any(np.abs(a) > self.amax + 1e-8):
            raise ValueError("reference exceeds velocity/acceleration limits")
        return q, v, a

    def sample(self, now_ns):
        if now_ns < self.sample_ns:
            raise ValueError("reference clock went backwards")
        q, v, a = self._state(*self.trajectory.at_time((now_ns - self.origin_ns) / 1e9))
        dt = (now_ns - self.sample_ns) / 1e9
        if (np.any(np.abs(q - self.q) > self.vmax * dt + 1e-8)
                or np.any(np.abs(v - self.v) > self.amax * dt + 1e-8)):
            raise ValueError("unsafe reference step")
        self.q, self.v, self.sample_ns = q.copy(), v.copy(), int(now_ns)
        return q.copy(), v.copy(), a.copy()

    def replace(self, goal, now_ns):
        from ruckig import Result, Trajectory

        goal = np.asarray(goal, dtype=np.float64)
        issue = check_arm_target(goal, joint_limit_lower_rad=self.lower,
                                 joint_limit_upper_rad=self.upper)
        if issue:
            raise ValueError(f"invalid q_goal: {issue}")
        if self.goal is not None:
            self.sample(now_ns)
            if np.array_equal(goal, self.goal):
                return  # Identical commands retain the existing trajectory and velocity.
        self.input.current_position = self.q.tolist()
        self.input.current_velocity = self.v.tolist()
        self.input.target_position = goal.tolist()
        self.otg.validate_input(self.input, True, True)
        trajectory = Trajectory(7)
        if self.otg.calculate(self.input, trajectory) not in (Result.Working, Result.Finished):
            raise ValueError("Ruckig calculation failed")
        # Community min/max_position are not a constraint solver. Check the COMPLETE
        # trajectory, including braking extrema between sends, before it can be used.
        for i, bounds in enumerate(trajectory.position_extrema):
            if (not np.isfinite([bounds.min, bounds.max]).all()
                    or bounds.min < self.lower[i] or bounds.max > self.upper[i]):
                raise ValueError(f"reference braking/position extremum outside joint {i + 1} limits")
        self.trajectory = trajectory
        self.origin_ns = int(now_ns)
        self.goal = goal.copy()

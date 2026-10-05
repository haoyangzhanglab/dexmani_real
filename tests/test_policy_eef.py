import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.commands import RobotCommand


def test_eef_condition_is_public_fk_of_frozen_joint():
    cfg = ExperimentConfig()
    q = np.array(cfg.arm.home_qpos)
    hand = np.deg2rad(cfg.hand.home_qpos_deg)
    command = RobotCommand(1, q, hand)
    realizer = ActionRealizer(cfg)
    condition = realizer.control_from_command(command, "eef")
    np.testing.assert_array_equal(condition[:9], compute_eef_pose_history_xarm_base(q[None])[0])
    np.testing.assert_array_equal(condition[9:], hand)


def test_real_ik_dynamic_checks_without_resolving():
    cfg = ExperimentConfig()
    realizer = ActionRealizer.for_mode(cfg, "eef")
    solver = realizer.planner.online_ik_solver
    q = np.array(cfg.arm.home_qpos)
    hand = np.deg2rad(cfg.hand.home_qpos_deg)
    command = RobotCommand(1, q, hand)
    assert realizer.frozen_is_valid(command, q, q, "eef")
    previous = q.copy()
    previous[1] += 1
    assert solver.dynamic_rejection(q, q, previous) == "jump"
    assert not realizer.frozen_is_valid(command, q, previous, "eef")
    current = q.copy()
    current[0] += 2 * np.pi
    assert solver.dynamic_rejection(q, current, q) == "band_switch"
    invalid = q.copy()
    invalid[1] = 1e4
    assert solver.dynamic_rejection(invalid, q, q) == "operational_limits"

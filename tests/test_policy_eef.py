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
    assert solver.dynamic_rejection(q, current, q) == "hw_dist"
    invalid = q.copy()
    invalid[1] = 1e4
    assert solver.dynamic_rejection(invalid, q, q) == "operational_limits"


def test_native_fk_ik_and_synthetic_table_path():
    from dexmani_real.planning.paths import _check_home_path_candidate

    cfg = ExperimentConfig()
    planner = ActionRealizer.for_mode(cfg, "eef").planner
    planner.set_hand_qpos(np.deg2rad(cfg.hand.home_qpos_deg))
    q = np.array(cfg.arm.home_qpos)
    pose = planner.kin.compute_eef_pose_world(q)
    result = planner.solve_online_ik(pose, q, q)
    assert result.success, result.reason
    np.testing.assert_allclose(planner.kin.compute_eef_pose_world(result.qpos).p, pose.p, atol=1e-6)
    # Synthetic planes bracket the whole robot. This exercises native geometry
    # and path validation only; it makes no claim about the physical desk.
    for table_height, safe in ((-10, True), (10, False)):
        candidate = _check_home_path_candidate(
            np.array([q, result.qpos]),
            "native_offline",
            planner,
            table_z_surface_m=table_height,
            hand_safety_margin_m=0,
            allow_table_soft_escape=False,
        )
        assert candidate.safe == safe
        if not safe:
            assert candidate.reason == "table_clearance"

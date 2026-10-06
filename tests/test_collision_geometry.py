"""Nominal collision assets and compensated physical FK; native models, no devices."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning.collision import CollisionModel
from dexmani_real.robot.action import ActionIntent, ActionRealizer
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.model import (
    HAND_SDK_TO_URDF_IDX,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_MODEL_DIR,
)

# Confirmed against the original assets below, independently of runtime loading.
COLLIDING_ARM = np.array(
    [
        1.1665669814,
        0.1618457586,
        -1.0397002599,
        0.0107616103,
        -3.6491049933,
        0.1015522605,
        -0.4494161931,
    ]
)


def original_model(hand_dof):
    path = XARM7_XHAND_RIGHT_URDF_PATH if hand_dof else XARM7_XHAND_COLLISION_URDF_PATH
    model = pin.buildModelFromUrdf(str(path))
    geometry = pin.buildGeomFromUrdf(
        model, str(path), pin.GeometryType.COLLISION, package_dirs=[str(XHAND_MODEL_DIR)]
    )
    # Match the existing SRDF-only exclusions. Pinocchio's addAllCollisionPairs
    # also skips geometries sharing a parent joint, especially in the fixed hand.
    for first, second in combinations(range(geometry.ngeoms), 2):
        geometry.addCollisionPair(pin.CollisionPair(first, second))
    pin.removeCollisionPairs(model, geometry, str(XARM7_XHAND_SRDF_PATH))
    return NS(
        model=model, data=model.createData(), geometry=geometry, data_geom=geometry.createData()
    )


@pytest.mark.parametrize("hand_dof", [False, True], ids=["fixed", "full"])
@pytest.mark.parametrize("obstacle_near_hand", [False, True], ids=["distant", "contact"])
def test_collision_matches_original_urdf(hand_dof, obstacle_near_hand):
    from hppfcl.hppfcl import Box

    cfg = ExperimentConfig()
    reference = original_model(hand_dof)
    home, hand = np.array(cfg.arm.home_qpos), np.deg2rad(cfg.hand.home_qpos_deg)
    full_home = np.r_[home, hand[list(HAND_SDK_TO_URDF_IDX)]] if hand_dof else home
    pin.framesForwardKinematics(reference.model, reference.data, full_home)
    hand_pose = reference.data.oMf[reference.model.getFrameId("right_hand_link")]
    center = hand_pose.translation.copy() if obstacle_near_hand else np.full(3, 10.0)
    box = dict(name="test", center_xyz_m=center, size_xyz_m=(0.2,) * 3, quat_wxyz=(1, 0, 0, 0))
    cm = CollisionModel(hand_dof=hand_dof, static_boxes=(box,))
    # Independent environment oracle: original meshes against one world-space box.
    environment = original_model(hand_dof).geometry
    environment.removeAllCollisionPairs()
    obstacle = environment.addGeometryObject(
        pin.GeometryObject("obstacle", 0, 0, Box(0.2, 0.2, 0.2), pin.SE3(np.eye(3), center))
    )
    for i in range(reference.geometry.ngeoms):
        environment.addCollisionPair(pin.CollisionPair(i, obstacle))
    environment_data = environment.createData()
    assert [(p.first, p.second) for p in cm._collision_model.collisionPairs] == [
        (p.first, p.second) for p in reference.geometry.collisionPairs
    ]
    states = [
        (home, hand),
        (COLLIDING_ARM, hand),
        (np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7]), np.linspace(0.01, 0.3, 12)),
        (np.array([-0.3, 0.4, -0.2, 0.5, 0.6, -0.4, 0.2]), np.linspace(0.2, 0.5, 12)),
    ]
    for index, (arm, hand) in enumerate(states):
        q = np.r_[arm, hand[list(HAND_SDK_TO_URDF_IDX)]] if hand_dof else arm
        expected = bool(
            pin.computeCollisions(
                reference.model, reference.data, reference.geometry, reference.data_geom, q, False
            )
        )
        if index < 2:
            assert expected == (index == 1)
        cm.set_hand_qpos(hand)
        assert cm.check_self_collision(arm) == expected
        details = cm.check_self_collision_details(arm)
        expected_pairs = {
            (
                reference.geometry.geometryObjects[pair.first].name,
                reference.geometry.geometryObjects[pair.second].name,
            )
            for i, pair in enumerate(reference.geometry.collisionPairs)
            if pin.computeCollision(reference.geometry, reference.data_geom, i)
        }
        assert details.in_collision == expected
        assert {
            (pair.object_name1, pair.object_name2) for pair in details.collision_pairs
        } == expected_pairs
        assert details.num_contacts == len(expected_pairs)
        expected_environment = bool(
            pin.computeCollisions(
                reference.model, reference.data, environment, environment_data, q, False
            )
        )
        if index == 0:
            assert expected_environment == obstacle_near_hand
        assert cm.check_environment_collision(arm) == expected_environment
        # All mesh placements (including fingers) must use the original model,
        # even when a boolean collision result alone would not expose a shift.
        for i in range(reference.geometry.ngeoms):
            for actual in (cm._collision_data, cm._environment_collision_data):
                np.testing.assert_allclose(
                    actual.oMg[i].homogeneous, reference.data_geom.oMg[i].homogeneous, atol=1e-12
                )
    empty = CollisionModel(hand_dof=hand_dof)
    empty.set_hand_qpos(hand)
    assert not empty.check_environment_collision(arm)
    assert empty._environment_collision_model is None


@pytest.mark.parametrize(
    "mount",
    [None, ((0.02, -0.03, 0.04), (0.3, -0.4, 0.5)), ((-0.03, 0.02, -0.01), (-0.2, 0.6, -0.4))],
    ids=["physical-default", "translated-rotated-1", "translated-rotated-2"],
)
def test_real_pose_compensation_is_independent_of_collision_and_planning(mount):
    from dexmani_real.dataset.contracts import ProcessingConfig
    from dexmani_real.dataset.processing import _kinematic_arrays
    from dexmani_real.deployment.observation import (
        build_observation_kinematics,
        build_policy_observation,
    )

    cfg = ExperimentConfig()
    assert cfg.hand.T_eef_handbase_pos_xyz == (-0.015, 0.0, 0.0)
    if mount is not None:
        xyz, rpy = mount
        quat = tuple(np.roll(Rotation.from_euler("xyz", rpy).as_quat(), 1))
        cfg = replace(
            cfg, hand=replace(cfg.hand, T_eef_handbase_pos_xyz=xyz, T_eef_handbase_quat_wxyz=quat)
        )
    info = NS(
        observation_fields=("eef_pose", "fingertip_points"),
        fingertip_link_names=XHAND_FINGERTIP_LINK_NAMES,
    )
    fk = build_observation_kinematics(info, cfg)
    processing = ProcessingConfig.from_runtime(cfg, table_plane_abcd=(0, 0, 1, 10))
    np.testing.assert_array_equal(
        processing.handbase_position_eef_m, cfg.hand.T_eef_handbase_pos_xyz
    )
    reference = original_model(True)
    planner = ActionRealizer.for_mode(cfg, "eef").planner
    joint = ActionRealizer.for_mode(cfg, "joint")
    assert Path(planner.mplib_planner.urdf) == XARM7_XHAND_COLLISION_URDF_PATH
    for arm, hand in (
        (np.array(cfg.arm.home_qpos), np.deg2rad(cfg.hand.home_qpos_deg)),
        (np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7]), np.linspace(0.01, 0.3, 12)),
    ):
        q = np.r_[arm, hand[list(HAND_SDK_TO_URDF_IDX)]]
        expected_collision = bool(
            pin.computeCollisions(
                reference.model, reference.data, reference.geometry, reference.data_geom, q, False
            )
        )
        pin.updateFramePlacements(reference.model, reference.data)
        eef = reference.data.oMf[reference.model.getFrameId("custom_eef_link")]
        nominal_hand = reference.data.oMf[reference.model.getFrameId("right_hand_link")]
        physical_hand = eef * pin.SE3(
            Rotation.from_quat(np.roll(cfg.hand.T_eef_handbase_quat_wxyz, -1)).as_matrix(),
            np.array(cfg.hand.T_eef_handbase_pos_xyz),
        )
        if mount is None:
            np.testing.assert_allclose(
                physical_hand.translation - nominal_hand.translation,
                eef.rotation @ np.array([-0.010, 0, 0]),
                atol=1e-12,
            )
        expected_tips = [
            physical_hand.act(
                nominal_hand.actInv(
                    reference.data.oMf[reference.model.getFrameId(name)].translation
                )
            )
            for name in XHAND_FINGERTIP_LINK_NAMES
        ]
        rows = [NS(arm={"qpos": arm[None]}, hand={"qpos": hand[None]})]
        observed = build_policy_observation(rows, info, kinematics=fk)
        poses, _, tips = _kinematic_arrays(arm[None], hand[None], arm[None], fk.hand_fk, processing)
        np.testing.assert_allclose(observed["fingertip_points"][0], expected_tips, atol=1e-6)
        np.testing.assert_allclose(tips[0], expected_tips, atol=1e-6)
        np.testing.assert_allclose(observed["eef_pose"], poses, atol=1e-6)
        np.testing.assert_allclose(poses[0, :3], eef.translation, atol=1e-12)
        for cm in (joint.collision_model, planner.collision_model):
            cm.set_hand_qpos(hand)
            assert cm.check_self_collision(arm) == expected_collision
            pin.updateFramePlacements(cm._model, cm._data)
            np.testing.assert_allclose(
                cm._data.oMf[cm._model.getFrameId("right_hand_link")].homogeneous,
                nominal_hand.homogeneous,
                atol=1e-12,
            )
        mp = planner.pinocchio_model
        mp.compute_forward_kinematics(arm)
        for name, expected_pose in (("right_hand_link", nominal_hand), ("custom_eef_link", eef)):
            pose = mp.get_link_pose(mp.get_link_names().index(name))
            np.testing.assert_allclose(pose.p, expected_pose.translation, atol=1e-6)
            np.testing.assert_allclose(
                Rotation.from_quat(np.roll(pose.q, -1)).as_matrix(),
                expected_pose.rotation,
                atol=1e-6,
            )


def test_original_collision_blocks_commands_and_planner_postcheck():
    from dexmani_real.planning.planner import XArm7MotionPlanner

    cfg = ExperimentConfig()
    hand = np.deg2rad(cfg.hand.home_qpos_deg)
    reference = original_model(True)
    assert pin.computeCollisions(
        reference.model,
        reference.data,
        reference.geometry,
        reference.data_geom,
        np.r_[COLLIDING_ARM, hand[list(HAND_SDK_TO_URDF_IDX)]],
        False,
    )
    realizer = ActionRealizer.for_mode(cfg, "joint")
    # Identical current/previous state isolates collision admission from jump checks.
    result = realizer.realize(
        ActionIntent("joint", COLLIDING_ARM, hand), COLLIDING_ARM, COLLIDING_ARM
    )
    assert result.arm_qpos is None and result.rejection_reason == "self_collision"
    assert not realizer.frozen_is_valid(
        RobotCommand(1, COLLIDING_ARM, hand), COLLIDING_ARM, COLLIDING_ARM, "joint"
    )
    assert realizer.rejection_reason == "self_collision"
    planner = XArm7MotionPlanner.create_default()
    planner.set_hand_qpos(hand)
    home = np.array(cfg.arm.home_qpos)
    assert not planner.mplib_planner.check_for_self_collision(home)
    assert planner.mplib_planner.check_for_self_collision(COLLIDING_ARM)
    start = home.copy()
    start[0] += 0.01
    path = planner.plan_joint_qpos_path(home, start)
    assert path.success, path.reason
    assert path.report["terminal_pos_error_m"] <= planner.planning_profile.max_pose_error_pos_m
    assert path.report["terminal_rot_error_rad"] <= planner.planning_profile.max_pose_error_rot_rad
    assert not any(planner.collision_model.check_self_collision(q) for q in path.qpos_path)
    # A reported planning success must still pass the full-hand posterior check.
    rejected = planner.result_from_mplib(
        {"status": "Success", "position": np.repeat(COLLIDING_ARM[None], 2, axis=0)},
        planner.kin.compute_eef_pose_world(COLLIDING_ARM),
        COLLIDING_ARM,
        source="offline_counterexample",
        profile=planner.planning_profile,
    )
    assert not rejected.success and rejected.report["path_collision"]

"""Pure geometry checks for fixed-calibration wrist-to-EEF mapping."""

from __future__ import annotations

import unittest

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.teleop.arm_mapper import ArmWristMapper


def _axis_angle_quat_wxyz(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    normalized_axis = np.asarray(axis, dtype=np.float64).copy()
    normalized_axis /= np.linalg.norm(normalized_axis)
    half_angle_rad = np.deg2rad(angle_deg) / 2.0
    return np.concatenate(
        (
            np.array([np.cos(half_angle_rad)], dtype=np.float64),
            normalized_axis * np.sin(half_angle_rad),
        )
    )


def _yaw_deg(quat_wxyz: np.ndarray) -> float:
    rotation = quat2mat(quat_wxyz)
    return float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _mapper(*, vr_to_robot_rot: np.ndarray | None = None) -> ArmWristMapper:
    mapper = ArmWristMapper(
        vr_to_robot_rot=vr_to_robot_rot,
        max_delta_rot_rad=np.deg2rad(179.0),
        max_per_frame_rot_rad=np.deg2rad(30.0),
    )
    identity_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    mapper.reset(
        wrist_pos=np.zeros(3),
        wrist_quat_wxyz=identity_quat_wxyz,
        eef_pos=np.zeros(3),
        eef_quat_wxyz=identity_quat_wxyz,
    )
    return mapper


class ArmWristMapperTest(unittest.TestCase):
    def test_rotation_gate_advances_from_accepted_pose(self) -> None:
        mapper = _mapper()
        wrist_quat_wxyz = _axis_angle_quat_wxyz(np.array([0.0, 0.0, 1.0]), 180.0)

        target_yaws_deg = []
        for _ in range(3):
            mapped = mapper.map(np.zeros(3), wrist_quat_wxyz)
            if mapped is None:
                self.fail("mapper unexpectedly held a valid pose")
            target_yaws_deg.append(_yaw_deg(mapped["quat_wxyz"]))

        np.testing.assert_allclose(target_yaws_deg, [30.0, 60.0, 90.0], atol=1e-6)

    def test_rotation_spike_recovers_without_a_delayed_large_target(self) -> None:
        mapper = _mapper()
        spike_quat_wxyz = _axis_angle_quat_wxyz(np.array([0.0, 0.0, 1.0]), 180.0)
        identity_quat_wxyz = _axis_angle_quat_wxyz(np.array([0.0, 0.0, 1.0]), 0.0)

        first = mapper.map(np.zeros(3), spike_quat_wxyz)
        second = mapper.map(np.zeros(3), identity_quat_wxyz)

        if first is None or second is None:
            self.fail("mapper unexpectedly held a valid pose")
        self.assertAlmostEqual(_yaw_deg(first["quat_wxyz"]), 30.0, places=6)
        self.assertAlmostEqual(_yaw_deg(second["quat_wxyz"]), 0.0, places=6)

    def test_fixed_calibration_maps_position_and_rotation_with_one_transform(
        self,
    ) -> None:
        vr_to_robot_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        mapper = _mapper(vr_to_robot_rot=vr_to_robot_rot)
        wrist_quat_wxyz = _axis_angle_quat_wxyz(np.array([1.0, 0.0, 0.0]), 30.0)

        mapped = mapper.map(np.array([1.0, 0.0, 0.0]), wrist_quat_wxyz)

        if mapped is None:
            self.fail("mapper unexpectedly held a valid pose")
        np.testing.assert_allclose(mapped["pos"], np.array([0.0, 1.0, 0.0]))
        expected_rot = vr_to_robot_rot @ quat2mat(wrist_quat_wxyz) @ vr_to_robot_rot.T
        np.testing.assert_allclose(
            quat2mat(mapped["quat_wxyz"]), expected_rot, atol=1e-12
        )

    def test_robot_world_x_rotation_remains_world_x_with_nonidentity_anchors(
        self,
    ) -> None:
        """A spatial robot-X wrist rotation must pre-multiply the EEF target."""
        vr_to_robot_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        wrist_quat0 = _axis_angle_quat_wxyz(np.array([0.0, 1.0, 0.0]), 40.0)
        eef_quat0 = _axis_angle_quat_wxyz(np.array([0.0, 0.0, 1.0]), -25.0)
        robot_world_x_delta = quat2mat(
            _axis_angle_quat_wxyz(np.array([1.0, 0.0, 0.0]), 20.0)
        )
        wrist_rot0 = quat2mat(wrist_quat0)
        wrist_rot = (
            vr_to_robot_rot.T @ robot_world_x_delta @ vr_to_robot_rot @ wrist_rot0
        )
        mapper = _mapper(vr_to_robot_rot=vr_to_robot_rot)
        mapper.reset(
            wrist_pos=np.zeros(3),
            wrist_quat_wxyz=wrist_quat0,
            eef_pos=np.zeros(3),
            eef_quat_wxyz=eef_quat0,
        )

        mapped = mapper.map(np.zeros(3), mat2quat(wrist_rot))

        if mapped is None:
            self.fail("mapper unexpectedly held a valid pose")
        np.testing.assert_allclose(
            quat2mat(mapped["quat_wxyz"]),
            robot_world_x_delta @ quat2mat(eef_quat0),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

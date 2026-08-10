from __future__ import annotations

import numpy as np
import pytest
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat

from dexmani_real.sensor.vr_receiver_process import _normalized_wxyz
from dexmani_real.teleop.arm_mapper import ArmWristMapper

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _reset_identity(mapper: ArmWristMapper) -> None:
    mapper.reset(np.zeros(3), _IDENTITY_QUAT, np.zeros(3), _IDENTITY_QUAT)
    assert mapper.is_ready()


@pytest.mark.parametrize("axis", np.eye(3))
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_per_frame_rotation_clamp_is_sign_symmetric(axis: np.ndarray, sign: float) -> None:
    limit = np.deg2rad(30.0)
    mapper = ArmWristMapper(max_delta_rot_rad=np.deg2rad(179.0), max_per_frame_rot_rad=limit)
    _reset_identity(mapper)

    wrist_quat = mat2quat(axangle2mat(axis, sign * np.deg2rad(60.0)))
    mapped = mapper.map(np.zeros(3), wrist_quat)

    assert mapped is not None
    _, angle = mat2axangle(quat2mat(mapped["quat_wxyz"]))
    assert abs(angle) == pytest.approx(limit)
    assert np.sign(angle) == sign


@pytest.mark.parametrize("axis", np.eye(3))
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_total_rotation_clamp_is_sign_symmetric(axis: np.ndarray, sign: float) -> None:
    limit = np.deg2rad(80.0)
    mapper = ArmWristMapper(max_delta_rot_rad=limit, max_per_frame_rot_rad=np.deg2rad(179.0))
    _reset_identity(mapper)

    wrist_quat = mat2quat(axangle2mat(axis, sign * np.deg2rad(120.0)))
    mapped = mapper.map(np.zeros(3), wrist_quat)

    assert mapped is not None
    _, angle = mat2axangle(quat2mat(mapped["quat_wxyz"]))
    assert abs(angle) == pytest.approx(limit)
    assert np.sign(angle) == sign


def test_reset_rejects_invalid_pose_and_clears_previous_anchor() -> None:
    mapper = ArmWristMapper()
    _reset_identity(mapper)

    mapper.reset(np.zeros(3), np.zeros(4), np.zeros(3), _IDENTITY_QUAT)

    assert not mapper.is_ready()
    assert mapper.map(np.zeros(3), _IDENTITY_QUAT) is None


@pytest.mark.parametrize(
    "position, quaternion",
    [
        (np.zeros(2), _IDENTITY_QUAT),
        (np.array([np.nan, 0.0, 0.0]), _IDENTITY_QUAT),
        (np.zeros(3), np.zeros(4)),
        (np.zeros(3), np.array([np.inf, 0.0, 0.0, 0.0])),
        (np.zeros(3), np.ones(3)),
    ],
)
def test_invalid_map_frame_holds_without_mutating_temporal_state(position: np.ndarray, quaternion: np.ndarray) -> None:
    mapper = ArmWristMapper()
    _reset_identity(mapper)
    last_rot = mapper._last_wrist_rot.copy()  # noqa: SLF001 - temporal-state invariant
    last_quat = mapper.last_quat_wxyz.copy()

    assert mapper.map(position, quaternion) is None
    np.testing.assert_allclose(mapper._last_wrist_rot, last_rot)  # noqa: SLF001
    np.testing.assert_allclose(mapper.last_quat_wxyz, last_quat)


def test_nonunit_and_negated_quaternions_map_to_the_same_pose() -> None:
    rotation = mat2quat(axangle2mat(np.array([0.0, 0.0, 1.0]), 0.2))
    first = ArmWristMapper()
    second = ArmWristMapper()
    _reset_identity(first)
    _reset_identity(second)

    first_mapped = first.map(np.array([0.1, -0.2, 0.3]), rotation * 4.0)
    second_mapped = second.map(np.array([0.1, -0.2, 0.3]), -rotation)

    assert first_mapped is not None and second_mapped is not None
    np.testing.assert_allclose(first_mapped["pos"], second_mapped["pos"])
    np.testing.assert_allclose(
        quat2mat(first_mapped["quat_wxyz"]),
        quat2mat(second_mapped["quat_wxyz"]),
    )


def test_vr_receiver_quaternion_boundary_normalizes_and_rejects_invalid_values() -> None:
    normalized = _normalized_wxyz([2.0, 0.0, 0.0, 0.0], "test_quat")
    np.testing.assert_allclose(normalized, _IDENTITY_QUAT)

    with pytest.raises(ValueError, match="norm is too small"):
        _normalized_wxyz(np.zeros(4), "test_quat")
    with pytest.raises(ValueError, match="finite array"):
        _normalized_wxyz([np.nan, 0.0, 0.0, 0.0], "test_quat")
    with pytest.raises(ValueError, match="finite array"):
        _normalized_wxyz(np.ones(3), "test_quat")


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 180.0])
def test_heading_rotates_position_into_ego_forward_without_rotating_orientation_axes(heading_deg: float) -> None:
    mapper = ArmWristMapper(max_delta_rot_rad=np.deg2rad(179.0), max_per_frame_rot_rad=np.deg2rad(179.0))
    heading_rad = np.deg2rad(heading_deg)
    heading_quat = mat2quat(axangle2mat(np.array([0.0, 0.0, 1.0]), heading_rad))
    mapper.set_heading(heading_quat)
    _reset_identity(mapper)

    operator_forward = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
    wrist_rotation = mat2quat(axangle2mat(np.array([0.0, 1.0, 0.0]), 0.2))
    mapped = mapper.map(operator_forward, wrist_rotation)

    assert mapped is not None
    np.testing.assert_allclose(mapped["pos"], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(quat2mat(mapped["quat_wxyz"]), quat2mat(wrist_rotation), atol=1e-12)

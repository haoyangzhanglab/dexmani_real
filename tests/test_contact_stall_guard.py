from __future__ import annotations

import numpy as np
import pytest

from dexmani_real.config.defaults import PolicyParams
from dexmani_real.policy.vr_teleop_policy import _contact_stall_detected


def _detected(
    *,
    command_error_rad: float = 0.20,
    closing_speed_rad_s: float = 0.0,
    eef_z_m: float = 0.126,
    target_z_m: float = 0.120,
) -> bool:
    qpos = np.zeros(7)
    command = np.zeros(7)
    command[1] = command_error_rad
    qvel = np.zeros(7)
    qvel[1] = closing_speed_rad_s
    return _contact_stall_detected(
        qpos,
        qvel,
        command,
        np.array([0.60, 0.18, eef_z_m]),
        np.array([0.60, 0.18, target_z_m]),
        table_z_surface_m=0.022,
        table_context_height_m=0.18,
        min_downward_target_m=0.003,
        tracking_error_rad=0.18,
        max_closing_speed_rad_s=0.05,
    )


def test_detects_blocked_downward_command_near_table() -> None:
    assert _detected()


def test_allows_downward_command_that_is_still_converging() -> None:
    assert not _detected(closing_speed_rad_s=0.10)


def test_allows_small_tracking_lag_and_upward_retreat() -> None:
    assert not _detected(command_error_rad=0.10)
    assert not _detected(target_z_m=0.132)


def test_table_height_is_context_not_an_exclusion_zone() -> None:
    assert not _detected(eef_z_m=0.25, target_z_m=0.24)
    assert not _detected(command_error_rad=0.0, eef_z_m=0.022, target_z_m=0.010)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_stall_table_context_height_m": 0.0},
        {"contact_stall_min_downward_target_m": 0.0},
        {"contact_stall_tracking_error_rad": 0.0},
        {"contact_stall_max_closing_speed_rad_s": -0.01},
    ],
)
def test_contact_stall_config_rejects_invalid_thresholds(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PolicyParams(**kwargs)

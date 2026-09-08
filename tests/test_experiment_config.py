"""Offline behavior checks for the single runtime configuration boundary."""

import pickle

import numpy as np
import pytest
import yaml

from dexmani_real.config.experiment import resolve_experiment_config


def test_yaml_and_cli_precedence_without_mutating_defaults(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("arm:\n  loop_hz: 24\npolicy:\n  recording_enabled: false\n")
    cfg = resolve_experiment_config(
        yaml_path=path, cli_overrides={"arm.loop_hz": 25, "policy.control_hz": None}
    )
    assert cfg.arm.loop_hz == 25
    assert cfg.policy.recording_enabled is False
    assert resolve_experiment_config().arm.loop_hz == 30
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.arm.loop_hz == 25
    assert restored.safety.heartbeat_timeouts == cfg.safety.heartbeat_timeouts
    cfg.safety.heartbeat_timeouts["arm"] = 99
    assert resolve_experiment_config().safety.heartbeat_timeouts["arm"] == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"unknown": {}},
        {"arm": {"unknown": 1}},
        {"policy": {"workspace": {"unknown": 1}}},
        {"environment": {"static_boxes": [{"unknown": 1}]}},
        {"safety": {"heartbeat_timeouts": {"unknown": 1}}},
        {"policy": {"hand_enabled": "false"}},
        {"environment": {"table": {"enabled": "false"}}},
    ],
)
def test_unknown_fields_and_ambiguous_external_booleans_fail(patch):
    with pytest.raises((TypeError, ValueError)):
        resolve_experiment_config(data=patch)


def test_calibration_is_loaded_or_rejected_without_fallback(tmp_path):
    path = tmp_path / "plane.json"
    path.write_text('{"a": 0, "b": 0, "c": 1, "d": -0.03}')
    data = {"environment": {"table": {"plane_path": str(path)}}}
    cfg = resolve_experiment_config(data=data)
    np.testing.assert_allclose(cfg.environment.table.plane_abcd, (0, 0, 1, -0.03))
    path.write_text('{"a": 0}')
    with pytest.raises(ValueError):
        resolve_experiment_config(data=data)


def test_hand_operational_and_mechanical_limits_remain_nested():
    cfg = resolve_experiment_config()
    for field, original in (
        ("qpos_min_rad", cfg.hand.mechanical_qpos_min_rad),
        ("mechanical_qpos_min_rad", cfg.hand.mechanical_qpos_min_rad),
    ):
        lower = list(original)
        lower[0] -= 0.01
        with pytest.raises(ValueError):
            resolve_experiment_config(data={"hand": {field: lower}})
    upper = list(cfg.hand.mechanical_qpos_max_rad)
    upper[0] += 0.01
    with pytest.raises(ValueError):
        resolve_experiment_config(data={"hand": {"qpos_max_rad": upper}})


def test_zero_width_joint_limits_and_home_outside_bounds_fail():
    cfg = resolve_experiment_config()
    lower = list(cfg.arm.joint_limit_lower)
    upper = list(cfg.arm.joint_limit_upper)
    lower[0] = upper[0] = cfg.arm.home_qpos[0]
    with pytest.raises(ValueError):
        resolve_experiment_config(
            data={"arm": {"joint_limit_lower": lower, "joint_limit_upper": upper}}
        )
    lower = list(cfg.hand.qpos_min_rad)
    upper = list(cfg.hand.qpos_max_rad)
    lower[0] = upper[0] = np.deg2rad(cfg.hand.home_qpos_deg[0])
    with pytest.raises(ValueError):
        resolve_experiment_config(
            data={"hand": {"qpos_min_rad": lower, "qpos_max_rad": upper}}
        )
    for patch in (
        {"arm": {"home_qpos": [100] * 7}},
        {"hand": {"home_qpos_deg": [1000] * 12}},
    ):
        with pytest.raises(ValueError):
            resolve_experiment_config(data=patch)


@pytest.mark.parametrize(
    "patch",
    [
        {"arm": {"loop_hz": 0}},
        {"policy": {"action_validity_s": 0.1, "command_progress_timeout_s": 0.2}},
        {"policy": {"workspace": {"x_min": 0.72, "x_max": 0.72}}},
        {"keyboard_teleop": {"workspace_command_margin_m": 10}},
        {"environment": {"static_boxes": [{"size_xyz_m": [1, 0, 1]}]}},
        {"environment": {"static_boxes": [{"quat_wxyz": [0, 0, 0, 0]}]}},
        {"environment": {"table": {"plane_path": None, "plane_abcd": [0, 0, -1, 0]}}},
        {"safety": {"heartbeat_timeouts": {"arm": 0}}},
        {"camera": {"max_frame_age_s": 3, "recording_stall_abort_s": 2}},
    ],
)
def test_geometry_and_timing_relations_fail_at_load(patch):
    with pytest.raises(ValueError):
        resolve_experiment_config(data=patch)


def test_valid_static_box_and_print_config_do_not_start_hardware(monkeypatch, capsys):
    from examples import collect_teleop

    cfg = resolve_experiment_config(
        data={
            "environment": {
                "static_boxes": [{"name": "fixture", "size_xyz_m": [1, 2, 3]}]
            }
        }
    )
    assert cfg.environment.static_boxes[0].name == "fixture"

    def unexpected_run(*args, **kwargs):
        pytest.fail("print-config must not start teleoperation")

    monkeypatch.setattr(collect_teleop, "run_teleop_experiment", unexpected_run)
    assert collect_teleop.main(["--print-config", "--no-record"]) == 0
    values = yaml.safe_load(capsys.readouterr().out)
    assert values["policy"]["recording_enabled"] is False
    assert "canonical_yaml" not in values
    assert "canonical_json" not in values

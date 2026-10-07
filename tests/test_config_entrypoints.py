"""Declared configuration and offline startup boundaries; no device imports."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from dexmani_real.config.experiment import (
    ExperimentConfig,
    config_as_dict,
    load_experiment_config,
    resolve_runtime_table,
)
from dexmani_real.config.pointcloud import PointCloudConfig


@pytest.mark.parametrize("home,cloud", [(False, False), (True, False), (False, True), (True, True)])
def test_one_table_snapshot_for_actual_consumers(tmp_path, monkeypatch, home, cloud):
    from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
    from dexmani_real.config import experiment
    from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig

    path = tmp_path / "plane.json"
    path.write_text(json.dumps(dict(a=0, b=0, c=1, d=-0.3)))
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "camera": dict(
                    serial="synthetic",
                    type="eye_to_hand",
                    pose=dict(position=[0, 0, 0], orientation=[1, 0, 0, 0]),
                )
            }
        )
    )
    cfg = load_experiment_config(
        data={"environment": {"table": {"enabled": home, "plane_path": str(path)}}}
    )
    # This is the saved recipe, deliberately different from the current default.
    recipe = PointCloudConfig(remove_table=cloud, num_points=32)
    reads = []
    native = experiment.resolve_table_plane
    monkeypatch.setattr(
        experiment,
        "resolve_table_plane",
        lambda table: (reads.append(table.plane_path), native(table))[1],
    )
    resolved = resolve_runtime_table(cfg, pointcloud=recipe)
    path.write_text(json.dumps(dict(a=0, b=0, c=1, d=-0.9)))
    worker = PointCloudWorkerConfig(
        pointcloud=recipe,
        camera_calibration=CameraExtrinsics(camera),
        table_plane_abcd=resolved.environment.table.plane_abcd,
    )
    assert len(reads) == int(home or cloud)
    assert worker.table_plane_abcd == ((0, 0, 1, -0.3) if cloud else None)
    if home or cloud:
        assert config_as_dict(resolved)["environment"]["table"]["plane_abcd"] == [0, 0, 1, -0.3]
    assert worker.pointcloud is recipe


def test_execution_precedence_zero_and_required_budgets():
    info = NS(control_dt_s=0.1, horizon=8, n_obs_steps=2, n_action_steps=3)
    cfg = load_experiment_config(
        data={
            "execution": dict(
                execution_mode="rtc",
                max_decision_age_s=2.0,
                max_wait_s=1.0,
                max_tick_lateness_s=0.02,
                prefetch_steps=2,
                rtc_guidance_cap=3.0,
            )
        },
        cli_overrides={"execution.rtc_guidance_cap": 0.0, "execution.max_wait_s": None},
    )
    assert cfg.execution.validate(info).rtc_guidance_cap == 0
    assert cfg.execution.max_wait_s == 1
    with pytest.raises(ValueError, match="explicit"):
        ExperimentConfig().execution.validate(info)


@pytest.fixture(params=["data", "yaml"])
def config_loader(request, tmp_path):
    def load(data, **kwargs):
        if request.param == "data":
            return load_experiment_config(data=data, **kwargs)
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(data))
        return load_experiment_config(yaml_path=path, **kwargs)

    return load


@pytest.mark.parametrize("device_name", [None, "configured_device"])
def test_hand_device_name_accepts_null_or_string(config_loader, device_name):
    cfg = config_loader({"hand": {"device_name": device_name}})
    assert cfg.hand.device_name == device_name
    cfg.hand.validate()


@pytest.mark.parametrize("device_name", [7, 1.5, True, [], {}])
def test_hand_device_name_rejects_other_types(config_loader, device_name):
    with pytest.raises(TypeError, match="hand.device_name"):
        config_loader({"hand": {"device_name": device_name}})


def test_nullable_strings_remain_explicit_and_unknown_fields_fail(config_loader):
    cfg = config_loader({"environment": {"table": {"plane_path": None}}})
    assert cfg.environment.table.plane_path is None
    with pytest.raises(TypeError, match="arm.ip"):
        config_loader({"arm": {"ip": None}})
    with pytest.raises(TypeError, match="unknown config fields"):
        config_loader({"hand": {"device_nam": None}})


def test_cli_none_keeps_configured_hand_device_name(config_loader):
    cfg = config_loader(
        {"hand": {"device_name": "configured_device"}},
        cli_overrides={"hand.device_name": None},
    )
    assert cfg.hand.device_name == "configured_device"


@pytest.mark.parametrize("recording", [False, True])
def test_teleop_session_injects_recorder_without_starting_io(tmp_path, monkeypatch, recording):
    from test_review_remediation import shared_state

    from dexmani_real.recording.recorder import AsyncEpisodeRecorder
    from dexmani_real.teleop import runner as runner_module
    from dexmani_real.teleop import session

    runtime = load_experiment_config(
        data={
            "policy": {"episodes_dir": str(tmp_path / "captures"), "recording_enabled": recording},
            "environment": {"table": {"plane_path": None}},
        }
    )
    shared = shared_state()
    seen = []
    monkeypatch.setattr(session, "load_vr_transform", lambda _: NS(transform=np.eye(3)))
    monkeypatch.setattr(session, "load_optional_camera_extrinsics", lambda _: None)
    monkeypatch.setattr(session.ActionRealizer, "for_mode", lambda *a: NS())
    monkeypatch.setattr(session, "_build_hand_retargeter", lambda _: None)
    monkeypatch.setattr(session, "build_home_planner", lambda _: None)
    monkeypatch.setattr(session.RuntimeChannels, "create", lambda **kw: shared)
    monkeypatch.setattr(session, "RuntimeSupervisor", lambda *a: NS(check=lambda: True))
    monkeypatch.setattr(session, "shutdown_local_runtime", lambda *a, **kw: True)
    monkeypatch.setattr(
        runner_module,
        "AudioFeedback",
        lambda: NS(wait_until_idle=lambda **kw: True, close=lambda: None),
    )

    def refuse_connection():
        raise RuntimeError("offline boundary")

    monkeypatch.setattr(
        session, "DexManiRobot", lambda *a, **kw: NS(connect=refuse_connection, stop_required=False)
    )
    native_runner = session.TeleopRunner

    def construct(*args, **kwargs):
        assert "recorder" in kwargs
        recorder = kwargs["recorder"]
        if recording:
            assert isinstance(recorder, AsyncEpisodeRecorder)
            assert recorder.data_dir == tmp_path / "captures" / "assembly"
            assert recorder.control_hz == runtime.teleop.control_hz
            assert recorder._rgb_shape == (runtime.camera.height, runtime.camera.width, 3)
            assert recorder._thread is None and not recorder.is_recording
        else:
            assert recorder is None
        owner = native_runner(*args, **kwargs)
        assert owner.recorder is recorder
        seen.append(owner)
        return owner

    monkeypatch.setattr(session, "TeleopRunner", construct)
    assert session.run_teleop_experiment(runtime, task_name="assembly") == 1
    assert len(seen) == 1
    assert not (tmp_path / "captures").exists()


@pytest.mark.parametrize(
    "data",
    [
        {"typo": {}},
        {"vr": []},
        {"arm": {"ip": 1}},
        {"execution": {"max_wait_s": "x"}},
        {"pointcloud": {"workspace": "x"}},
        {"camera": {"l515_confidence_threshold": True}},
    ],
)
def test_declarations_reject_uninterpretable_fields(data):
    with pytest.raises((TypeError, ValueError)):
        load_experiment_config(data=data)


def test_unused_values_do_not_block_policy_and_direct_cloud_still_rejects():
    from dexmani_real.dataset.contracts import ProcessingConfig
    from dexmani_real.deployment.config import validate_policy_runtime_compatibility
    from dexmani_real.sensor.pointcloud import build_point_cloud

    cfg = load_experiment_config(
        data={"vr": {"port": -1}, "pointcloud": {"depth_min_m": -1}, "teleop": {"control_hz": -1}}
    )
    info = NS(control_dt_s=0.1, observation_fields=("joint_state",), action_mode="joint")
    assert validate_policy_runtime_compatibility(info, cfg) is None
    with pytest.raises(ValueError, match="depth range"):
        ProcessingConfig(pointcloud=cfg.pointcloud, table_plane_abcd=(0, 0, 1, 0))
    with pytest.raises(ValueError, match="depth range"):
        build_point_cloud(
            depth_raw=None,
            color=None,
            depth_scale_m=0.001,
            geometry=None,
            T_xarm_base_from_color=np.eye(4),
            table_plane_abcd=(0, 0, 1, 0),
            config=cfg.pointcloud,
        )
    recipe = PointCloudConfig.from_dict(PointCloudConfig().to_dict())
    assert recipe == PointCloudConfig()
    assert "outlier_candidate_multiplier" in recipe.to_dict()


@pytest.mark.parametrize(
    "script,args",
    [
        (name, ["--help"])
        for name in (
            "collect_teleop",
            "run_policy",
            "keyboard_teleop",
            "calibrate_camera",
            "calibrate_vr_heading",
            "replay_episode",
            "pointcloud_process_example",
            "realsense_record_example",
            "xhand_diagnostics",
            "export_policy_zarr",
            "visualize_episode",
        )
    ]
    + [
        ("collect_teleop", ["--print-config"]),
        ("run_policy", ["unused/task/experiment", "--print-config"]),
    ],
)
def test_help_and_declared_config_without_backends(script, args):
    program = """
import importlib.abc, pathlib, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nlopt','pynput','pyrealsense2','xarm','xhand','xhand_control','hand_tracking_sdk','torch','mplib','pinocchio','open3d','rerun'}:
            raise AssertionError('unexpected backend import: ' + fullname)
sys.meta_path.insert(0, Block())
original = pathlib.Path.open
def open_path(self, *a, **kw):
    if 'calibration/state' in str(self):
        raise AssertionError('unexpected calibration read: ' + str(self))
    return original(self, *a, **kw)
pathlib.Path.open = open_path
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", program, str(root / "examples" / f"{script}.py"), *args],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_teleop_constructs_before_allocating_or_connecting(monkeypatch):
    from dexmani_real.teleop import session

    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        environment=replace(cfg.environment, table=replace(cfg.environment.table, plane_path=None)),
    )
    monkeypatch.setattr(session, "load_vr_transform", lambda _: NS(transform=np.eye(3)))
    monkeypatch.setattr(
        session.RuntimeChannels, "create", lambda **kw: pytest.fail("allocated before construction")
    )

    def fail(*args):
        raise ValueError("selected model unavailable")

    monkeypatch.setattr(session.ActionRealizer, "for_mode", fail)
    with pytest.raises(ValueError, match="selected model unavailable"):
        session.run_teleop_experiment(cfg)


def test_direct_realizer_rejects_invalid_used_limits_and_workspace():
    from dexmani_real.robot.action import ActionRealizer

    cfg = ExperimentConfig()
    bad = replace(cfg, arm=replace(cfg.arm, joint_limit_lower=(float("nan"),) * 7))
    with pytest.raises(ValueError, match="arm action limits"):
        ActionRealizer(bad)
    bad = replace(cfg, policy=replace(cfg.policy, workspace=replace(cfg.policy.workspace, x_min=1)))
    with pytest.raises(ValueError, match="workspace"):
        ActionRealizer(bad, planner=NS())

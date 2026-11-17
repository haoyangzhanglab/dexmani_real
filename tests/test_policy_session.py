"""部署入口、启动取消与资源生命周期；模型和设备边界使用替身。"""

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from dexmani_policy.deployment.metadata import PolicyInfo

from dexmani_real.config.deployment import DEFAULT_MAX_DURATION_S, DEFAULT_NUM_EPISODES
from dexmani_real.config.real import RealConfig
from dexmani_real.data.raw import camera_metadata, robot_recording_parameters
from dexmani_real.deployment import session
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def policy_info():
    return PolicyInfo(
        checkpoint_path="policy.pt", weights="raw", inference_steps=4, task_name="test",
        observation_fields=("joint_state",), n_obs_steps=2, n_action_steps=4, future_steps=8,
        control_action_dim=19, action_mode="joint", rgb_preprocessing={}, supports_rtc=False,
        requires_prev_action=False, control_dt_s=.05, data_processing=None,
    )


@pytest.fixture
def policy_session(monkeypatch):
    monkeypatch.setattr(session, "SessionState", lambda: SessionState(stop_signal=Event(), emergency_signal=Event()))
    configured = session.PolicySession(RealConfig(), "policy.pt", device="cpu")
    configured.keyboard.start = Mock()
    configured.keyboard.stop = Mock()
    configured.keyboard.check_health = Mock(return_value=True)
    return configured


@pytest.mark.parametrize("name,value,error", [
    ("max_duration_s", True, TypeError),
    ("max_duration_s", 0, ValueError),
    ("max_duration_s", -1, ValueError),
    ("max_duration_s", float("nan"), ValueError),
    ("max_duration_s", float("inf"), ValueError),
    ("num_episodes", True, TypeError),
    ("num_episodes", 1.0, TypeError),
    ("num_episodes", 0, ValueError),
    ("num_episodes", -1, ValueError),
])
def test_invalid_budget_rejected_before_session_resources(monkeypatch, name, value, error):
    state = Mock(side_effect=AssertionError("must reject before allocating session state"))
    monkeypatch.setattr(session, "SessionState", state)
    with pytest.raises(error, match=name):
        session.PolicySession(RealConfig(), "policy.pt", **{name: value})
    state.assert_not_called()


def test_session_defaults_and_explicit_unbounded_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "SessionState", Mock(return_value=Mock(cancel_epoch=1)))
    runtime = RealConfig()
    configured = session.PolicySession(runtime, "policy.pt")
    assert configured.runtime is runtime and configured.policy == "policy.pt"
    assert configured.max_duration_s == DEFAULT_MAX_DURATION_S
    assert configured.num_episodes == DEFAULT_NUM_EPISODES
    assert configured.weights is configured.inference_steps is None
    assert configured.device == "cuda:0" and configured.seed == 0
    assert not configured.record_raw and configured.output_root is None
    assert configured.model is configured.robot is None
    assert configured.keyboard.listener is None and configured.sensors.processes == []

    monkeypatch.chdir(tmp_path)
    unlimited = session.PolicySession(
        runtime, "policy.pt", num_episodes=None, max_duration_s=None,
        camera_calibration_path="camera.json", record_raw=True,
    )
    assert unlimited.max_duration_s is unlimited.num_episodes is None
    assert unlimited.camera_calibration_path == tmp_path / "camera.json"
    assert unlimited.output_root == Path(session.__file__).resolve().parents[2] / "rollouts"


def load_cli(name="run_policy.py"):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("run_policy_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_calls_session_directly(monkeypatch, tmp_path):
    constructor = Mock()
    constructor.return_value.run.return_value = 7
    monkeypatch.setattr(session, "PolicySession", constructor)
    cli = load_cli()
    assert cli.main([
        "policy.pt", "--weights", "raw", "--inference-steps", "3", "--device", "cpu",
        "--seed", "9", "--num-episodes", "2", "--max-duration", "1.5", "--record-raw",
        "--output", str(tmp_path), "--camera-calibration", "camera.json",
        "--task-dir", "../task-assets",
    ]) == 7
    args, kwargs = constructor.call_args
    assert args == (RealConfig(), "policy.pt")
    assert kwargs == {
        "profiler": None, "weights": "raw", "inference_steps": 3, "device": "cpu", "seed": 9,
        "num_episodes": 2, "max_duration_s": 1.5, "record_raw": True, "output_root": tmp_path,
        "camera_calibration_path": Path("camera.json"),
        "task_dir": Path("../task-assets"),
    }
    constructor.return_value.run.assert_called_once_with()


def test_print_config_does_not_create_session(monkeypatch, capsys):
    constructor = Mock(side_effect=AssertionError("print-config must not create a session"))
    monkeypatch.setattr(session, "PolicySession", constructor)
    assert load_cli().main(["--print-config", "--task-dir", "/missing/task-assets"]) == 0
    assert "control_dt_s=None" in capsys.readouterr().out
    constructor.assert_not_called()


@pytest.mark.parametrize("task_dir", [None, "missing", "bad-template"])
def test_hoi_task_errors_precede_robot_creation(policy_session, policy_info, monkeypatch, tmp_path, task_dir):
    monkeypatch.chdir(tmp_path)
    if task_dir == "bad-template":
        (tmp_path / task_dir / "hoi").mkdir(parents=True)
        (tmp_path / task_dir / "hoi/template.npz").write_bytes(b"invalid template")
    policy_session.task_dir = task_dir
    policy_session.wait_model = lambda: replace(policy_info, observation_fields=("object_pointcloud",))
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    robot = Mock(side_effect=AssertionError("task errors must precede Robot creation"))
    monkeypatch.setattr(session, "Robot", robot)
    expected = ValueError if task_dir is None or task_dir == "bad-template" else FileNotFoundError
    with pytest.raises(expected):
        policy_session.prepare_resources()
    robot.assert_not_called()


def test_non_hoi_policy_does_not_read_task_assets(policy_session, policy_info, monkeypatch):
    policy_session.task_dir = object()  # 任何 Path 转换或文件读取都会失败。
    policy_session.wait_model = lambda: policy_info
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    monkeypatch.setattr(session, "Robot", Mock())
    policy_session.prepare_resources()
    assert policy_session.hoi_inputs is None


@pytest.mark.parametrize("field,ring_name", [("point_cloud", "pointcloud_ring"), ("object_pointcloud", "hoi_ring")])
@pytest.mark.parametrize("include_rgb", [False, True])
def test_policy_derived_channel_carries_only_requested_rgb(
    policy_session, policy_info, monkeypatch, tmp_path, field, ring_name, include_rgb,
):
    runtime = policy_session.runtime
    policy_session.runtime = replace(runtime, camera=replace(runtime.camera, width=4, height=3))
    policy_session.record_raw = True
    policy_session.output_root = tmp_path
    fields = (field, "rgb") if include_rgb else (field,)
    policy_session.wait_model = lambda: replace(policy_info, observation_fields=fields)
    monkeypatch.setattr(session, "resolve_hoi_inputs", Mock())
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    monkeypatch.setattr(session, "Robot", Mock())
    policy_session.prepare_resources()

    channels = policy_session.sensors.channels
    ring = getattr(channels, ring_name)
    assert channels.camera_ring.field_names == {"timestamp_ns", "rgb", "depth"}
    assert "depth" not in ring.field_names
    assert ("rgb" in ring.field_names) is include_rgb
    if include_rgb:
        assert ring.dtype["rgb"] == channels.camera_ring.dtype["rgb"]


@pytest.mark.parametrize("explicit,dt,period_ns", [
    (False, 5., 5_000_000_000),
    (True, 5., 5_000_000_000),
    (True, 2. - .25e-9, 2_000_000_000),
])
def test_resolved_period_must_fit_policy_lease(policy_session, policy_info, monkeypatch, explicit, dt, period_ns):
    policy_info = replace(policy_info, observation_fields=("object_pointcloud",))
    if explicit:
        runtime = policy_session.runtime
        policy_session.runtime = replace(runtime, execution=replace(runtime.execution, control_dt_s=dt))
    else:
        policy_info = replace(policy_info, control_dt_s=dt)
    policy_session.wait_model = lambda: policy_info
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    robot = Mock(side_effect=AssertionError("unexpected device allocation"))
    monkeypatch.setattr(session, "Robot", robot)
    with pytest.raises(ValueError, match=rf"execution.control_dt_s.*{period_ns} ns.*command lease.*2000000000 ns"):
        policy_session.prepare_resources()
    robot.assert_not_called()


@pytest.mark.parametrize("path_form", ["relative", "home"])
def test_hoi_session_uses_explicit_task_and_records_path(
    policy_session, policy_info, monkeypatch, tmp_path, path_form,
):
    task = tmp_path / "assets/pick"
    (task / "hoi").mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"objects": [{"name": "apple"}]}))
    np.savez(task / "hoi/template.npz", points_local=np.zeros((1024, 3), np.float32),
             offsets=np.array([0, 1024]), object_names=np.array(["apple"]))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    policy_session.task_dir = "assets/pick" if path_form == "relative" else "~/assets/pick"
    policy_session.output_root = tmp_path / "results"
    policy_session.wait_model = lambda: replace(
        policy_info, observation_fields=("object_pointcloud",), task_name="independent/experiment label",
    )
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    monkeypatch.setattr(session, "SensorChannels", Mock())
    robot = Mock()
    monkeypatch.setattr(session, "Robot", robot)
    policy_session.prepare_resources()
    assert policy_session.hoi_inputs[0] == task
    np.testing.assert_array_equal(policy_session.hoi_inputs[1]["object_names"], ["apple"])
    robot.return_value.connect.assert_not_called()
    policy_session.save_run_config()
    import yaml

    saved = yaml.safe_load((policy_session.output_dir / "run_config.yaml").read_text())
    assert saved["task_dir"] == str(task)


def test_cli_accepts_unbounded_duration(monkeypatch):
    constructor = Mock()
    constructor.return_value.run.return_value = 0
    monkeypatch.setattr(session, "PolicySession", constructor)
    assert load_cli().main(["policy.pt", "--max-duration", "none"]) == 0
    assert constructor.call_args.kwargs["max_duration_s"] is None


@pytest.mark.parametrize("script", ["run_policy.py", "collect_teleop.py"])
def test_declared_config_does_not_require_json_values(tmp_path, capsys, script):
    path = tmp_path / "experiment.py"
    path.write_text(
        "from dataclasses import replace\n"
        "from dexmani_real.config.real import RealConfig\n"
        "base = RealConfig()\n"
        "CONFIG = replace(base, arm=replace(base.arm, homing=object()), "
        "hand=replace(base.hand, home_tor_max=object()), "
        "execution=replace(base.execution, rtc_guidance_cap=object()))\n"
    )
    assert load_cli(script).main(["--config", str(path), "--print-config"]) == 0
    assert "object at" in capsys.readouterr().out


@pytest.mark.parametrize("action_mode", ["joint", "eef"])
@pytest.mark.parametrize("record_raw", [False, True])
def test_policy_records_only_used_parameters(monkeypatch, tmp_path, policy_info, action_mode, record_raw):
    runtime = RealConfig()
    runtime = replace(runtime, arm=replace(runtime.arm, homing=object()),
                      hand=replace(runtime.hand, home_tor_max=object()),
                      execution=replace(runtime.execution, rtc_guidance_cap=object(), prefetch_steps=object()),
                      environment=replace(runtime.environment, home_model_confirmed=object(), static_boxes=(object(),)))
    monkeypatch.setattr(session, "SessionState", lambda: SessionState(stop_signal=Event(), emergency_signal=Event()))
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    monkeypatch.setattr(session, "SensorChannels", Mock())
    policy_info = replace(
        policy_info, action_mode=action_mode, control_action_dim=19 if action_mode == "joint" else 21,
    )
    configured = session.PolicySession(
        runtime, "policy.pt", device="cpu", seed=7, max_duration_s=12,
        output_root=tmp_path, record_raw=record_raw,
    )
    configured.keyboard = Mock(check_health=Mock(return_value=True))
    configured.wait_model = lambda: policy_info
    configured.prepare_resources()
    parameters = configured.effective_parameters
    expected_policy = asdict(policy_info)
    expected_policy.pop("data_processing")
    expected = {
        **robot_recording_parameters(configured.runtime),
        "policy": expected_policy, "seed": 7, "device": "cpu", "max_duration_s": 12.,
        "execution": {"mode": "sync", "control_dt_s": .05, "action_steps": 4},
    }
    if action_mode == "eef":
        expected.update(
            motion=asdict(runtime.motion),
            environment={
                "workspace": asdict(runtime.environment.workspace),
                "eef_workspace_tolerance_m": runtime.environment.eef_workspace_tolerance_m,
            },
        )
    effective_camera = None
    if record_raw:
        expected["camera_config"] = asdict(runtime.camera)
        configured.camera_snapshot = CameraSnapshot(
            "camera-123", CameraIntrinsics(16, 16, 20., 20., 8., 8., "none", (0.,) * 5), .001, None,
        )
        effective_camera = {"camera_T_xarm_base_from_color": None, **camera_metadata(configured.camera_snapshot)}
    assert parameters == expected
    configured.save_run_config()
    import yaml

    saved = yaml.safe_load((configured.output_dir / "run_config.yaml").read_text())
    assert saved["effective_parameters"] == json.loads(json.dumps(expected, allow_nan=False))
    assert saved["effective_camera"] == effective_camera
    assert saved["training_processing"] is None
    assert saved["actual_processing"] == {}
    assert saved["task_dir"] is None
    assert "data_processing" not in saved


@pytest.mark.parametrize("port", [None, 8124])
def test_vr_calibration_uses_config_and_explicit_port(tmp_path, capsys, port):
    path = tmp_path / "experiment.py"
    path.write_text(
        "from dataclasses import replace\n"
        "from dexmani_real.config.real import RealConfig\n"
        "base = RealConfig()\n"
        "CONFIG = replace(base, vr=replace(base.vr, host='127.0.0.2', port=8123))\n"
    )
    cli = load_cli("calibrate_vr_heading.py")
    argv = ["--config", str(path), "--print-config"]
    if port is not None:
        argv += ["--port", str(port)]
    assert cli.main(argv) == 0
    output = capsys.readouterr().out
    assert "host='127.0.0.2'" in output
    assert f"port={8123 if port is None else port}" in output


def test_model_load_failure_precedes_device_allocation(policy_session, monkeypatch):
    from dexmani_policy import deployment

    failure = ValueError("checkpoint load failed")
    monkeypatch.setattr(deployment, "load_policy", Mock(side_effect=failure))
    robot = Mock()
    channels = Mock()
    monkeypatch.setattr(session, "Robot", robot)
    monkeypatch.setattr(session, "SensorChannels", channels)
    configured = policy_session
    try:
        with pytest.raises(ValueError) as caught:
            configured.prepare_resources()
        assert caught.value is failure
        robot.assert_not_called()
        channels.assert_not_called()
        assert configured.robot is None and configured.episode is None
        assert configured.sensors.processes == []
    finally:
        exit_code = configured.close(failure=failure)
    assert exit_code == 1
    assert configured.model.close_future.done()
    configured.keyboard.stop.assert_called_once()


def test_sensor_startup_cancel_reaches_session_cleanup(policy_session, policy_info, monkeypatch):
    from dexmani_policy import deployment

    monkeypatch.setattr(session, "configure_logging", lambda: None)
    configured = policy_session
    info = replace(policy_info, observation_fields=("rgb",))
    model = Mock(info=info)
    monkeypatch.setattr(deployment, "load_policy", Mock(return_value=model))
    robot = Mock()
    monkeypatch.setattr(session, "Robot", Mock(return_value=robot))
    channels = SimpleNamespace(camera_ring=SimpleNamespace(latest_sequence=0), is_running=SimpleNamespace(value=True))
    monkeypatch.setattr(session, "SensorChannels", Mock(return_value=channels))
    process = Mock(pid=123, exitcode=None)
    process.name = "camera"
    process.is_alive.return_value = True

    def start():
        assert configured.startup_allowed()
        configured.runtime_state.request_stop()
        assert not configured.startup_allowed()

    def join(**kwargs):
        assert process in configured.sensors.processes
        process.is_alive.return_value = False
        process.exitcode = 0

    process.start.side_effect = start
    process.join.side_effect = join
    context = SimpleNamespace(Process=Mock(return_value=process))
    monkeypatch.setattr(session.mp, "get_context", lambda *args: context)
    loop = Mock(wraps=configured.run_control_loop)
    monkeypatch.setattr(configured, "run_control_loop", loop)
    snapshot = Mock()
    monkeypatch.setattr(session, "snapshot_camera_parameters", snapshot)
    assert configured.run() == 0
    assert configured.summary["termination_reason"] == "startup_cancelled"
    process.start.assert_called_once()
    process.join.assert_called_once()
    assert configured.sensors.processes == [] and not channels.is_running.value
    robot.connect.assert_called_once()
    robot.close.assert_called_once()
    robot.send_action.assert_not_called()
    loop.assert_not_called()
    snapshot.assert_not_called()
    model.warmup.assert_not_called()
    model.predict.assert_not_called()
    model.close.assert_called_once()
    assert configured.episode is None and configured.episodes == []

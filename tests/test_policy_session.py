"""部署入口、启动取消与资源生命周期；模型和设备边界使用替身。"""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dexmani_policy.deployment.metadata import PolicyInfo

from dexmani_real.config.deployment import DEFAULT_MAX_RUNNING_S, DEFAULT_NUM_EPISODES
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment import session
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
    ("max_running_s", True, TypeError),
    ("max_running_s", 0, ValueError),
    ("max_running_s", -1, ValueError),
    ("max_running_s", float("nan"), ValueError),
    ("max_running_s", float("inf"), ValueError),
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
    assert configured.max_running_s == DEFAULT_MAX_RUNNING_S
    assert configured.num_episodes == DEFAULT_NUM_EPISODES
    assert configured.weights is configured.inference_steps is None
    assert configured.device == "cuda:0" and configured.seed == 0
    assert not configured.record_raw and configured.output_root is None
    assert configured.model is configured.robot is None
    assert configured.keyboard.listener is None and configured.sensors.processes == []

    monkeypatch.chdir(tmp_path)
    unlimited = session.PolicySession(
        runtime, "policy.pt", num_episodes=None, max_running_s=None,
        camera_calibration_path="camera.json", record_raw=True,
    )
    assert unlimited.max_running_s is unlimited.num_episodes is None
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
    ]) == 7
    args, kwargs = constructor.call_args
    assert args == (RealConfig(), "policy.pt")
    assert kwargs == dict(
        weights="raw", inference_steps=3, device="cpu", seed=9,
        num_episodes=2, max_running_s=1.5, record_raw=True, output_root=tmp_path,
        camera_calibration_path=Path("camera.json"),
    )
    constructor.return_value.run.assert_called_once_with()


def test_print_config_does_not_create_session(monkeypatch, capsys):
    constructor = Mock(side_effect=AssertionError("print-config must not create a session"))
    monkeypatch.setattr(session, "PolicySession", constructor)
    assert load_cli().main(["--print-config"]) == 0
    assert "control_dt_s=None" in capsys.readouterr().out
    constructor.assert_not_called()


def test_cli_accepts_unbounded_duration(monkeypatch):
    constructor = Mock()
    constructor.return_value.run.return_value = 0
    monkeypatch.setattr(session, "PolicySession", constructor)
    assert load_cli().main(["policy.pt", "--max-duration", "none"]) == 0
    assert constructor.call_args.kwargs["max_running_s"] is None


@pytest.mark.parametrize("script", ["run_policy.py", "collect_teleop.py", "tools/control_xhand.py"])
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


def test_policy_records_only_used_parameters(monkeypatch, tmp_path, policy_info):
    runtime = RealConfig()
    runtime = replace(runtime, arm=replace(runtime.arm, homing=object()),
                      hand=replace(runtime.hand, home_tor_max=object()),
                      execution=replace(runtime.execution, rtc_guidance_cap=object(), prefetch_steps=object()),
                      environment=replace(runtime.environment, home_model_confirmed=object(), static_boxes=(object(),)))
    monkeypatch.setattr(session, "SessionState", lambda: SessionState(stop_signal=Event(), emergency_signal=Event()))
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    configured = session.PolicySession(runtime, "policy.pt", device="cpu", output_root=tmp_path)
    configured.keyboard = Mock(check_health=Mock(return_value=True))
    configured.wait_model = lambda: policy_info
    configured.prepare_resources()
    parameters = configured.effective_parameters
    assert parameters["execution"] == dict(mode="sync", control_dt_s=.05, action_steps=4)
    assert "homing" not in parameters["arm"] and "home_tor_max" not in parameters["hand"]
    assert "environment" not in parameters and "camera_config" not in parameters
    assert "data_processing" not in parameters["policy"]
    json.dumps(parameters, allow_nan=False)
    configured.save_run_config()
    import yaml

    saved = yaml.safe_load((configured.output_dir / "run_config.yaml").read_text())
    assert saved["training_processing"] is None
    assert saved["actual_processing"] == {}
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
    import dexmani_policy.deployment as deployment

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
    import dexmani_policy.deployment as deployment

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

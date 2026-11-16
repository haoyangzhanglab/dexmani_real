"""部署入口的预算、默认值与 CLI 转发；不加载模型或连接设备。"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from dexmani_real.config.deployment import DEFAULT_MAX_RUNNING_S, DEFAULT_NUM_EPISODES
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment import session


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


def load_cli():
    path = Path(__file__).parents[1] / "scripts/run_policy.py"
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
    declared = json.loads(capsys.readouterr().out)
    assert declared["execution"]["control_dt_s"] is None
    constructor.assert_not_called()

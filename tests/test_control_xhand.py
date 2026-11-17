import signal
from dataclasses import replace
from unittest.mock import Mock

import pytest

from scripts.tools import control_xhand as diagnostic


def test_print_config_skips_hardware(monkeypatch, capsys):
    runtime = diagnostic.load_config(None)
    hand = replace(runtime.hand, home_tor_max=object())
    monkeypatch.setattr(diagnostic, "load_config", lambda path: replace(runtime, hand=hand))
    hardware = Mock()
    core_limit = Mock()
    monkeypatch.setattr(diagnostic, "run_hardware_session", hardware)
    monkeypatch.setattr(diagnostic.resource, "setrlimit", core_limit)

    assert diagnostic.main(["--print-config"]) == 0
    assert capsys.readouterr().out.strip() == str(hand)
    hardware.assert_not_called()
    core_limit.assert_not_called()


@pytest.mark.parametrize("outcome,code,actions", [
    ("complete", 0, 6),
    ("interrupted", 128 + signal.SIGINT, 1),
    ("read_interrupted", 128 + signal.SIGTERM, 0),
    ("connect_error", 2, 0),
    ("read_only", 0, 0),
])
def test_xhand_diagnostic_cleanup(monkeypatch, outcome, code, actions):
    hand_type = diagnostic.XHandDiagnostic
    status = Mock()
    if outcome == "read_interrupted":
        status.side_effect = lambda finger_id: signal.raise_signal(signal.SIGTERM)
    core_limit = Mock()
    before = [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
    events = []
    stop_action = hand_type.stop_action

    def connect(self, *, cancel_requested):
        core_limit.assert_called_once_with(diagnostic.resource.RLIMIT_CORE, (0, 0))
        if outcome == "connect_error":
            raise RuntimeError("connection failed")

    def execute(self, target):
        events.append("action")
        self.motion_pending = True
        if outcome == "interrupted":
            signal.raise_signal(signal.SIGINT)

    def cleanup(self):
        events.append("stop_action")
        stop_action(self)

    monkeypatch.setattr(diagnostic, "configure_logging", Mock())
    monkeypatch.setattr(diagnostic, "validate_serial_port", Mock())
    monkeypatch.setattr(diagnostic.resource, "setrlimit", core_limit)
    monkeypatch.setattr(hand_type, "connect", connect)
    monkeypatch.setattr(hand_type, "print_device_info", Mock())
    monkeypatch.setattr(hand_type, "print_state", status)
    monkeypatch.setattr(hand_type, "execute_action", execute)
    monkeypatch.setattr(hand_type, "dwell", Mock())
    monkeypatch.setattr(hand_type, "stop_action", cleanup)
    monkeypatch.setattr(hand_type, "get_state", Mock(return_value=None))
    monkeypatch.setattr(hand_type, "stop", lambda self, *args, **kwargs: events.append("stop"))
    monkeypatch.setattr(hand_type, "disconnect", lambda self: events.append("disconnect"))
    if outcome == "read_only":
        monkeypatch.setattr(diagnostic, "action_sequence", Mock(side_effect=AssertionError))

    args = ["--finger-id", "2"]
    if outcome == "read_only":
        args.append("--read-only")
    assert diagnostic.main(args) == code
    expected_cleanup = ["stop_action", "stop", "disconnect"] if actions else ["stop_action", "disconnect"]
    assert events == ["action"] * actions + expected_cleanup
    if outcome == "connect_error":
        status.assert_not_called()
    else:
        status.assert_called_once_with(2)
    assert [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)] == before

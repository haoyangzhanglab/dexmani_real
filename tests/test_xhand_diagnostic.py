from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import FeedbackUnavailable
from dexmani_real.robot.xhand import driver
from dexmani_real.robot.xhand.feedback import XHandState
from scripts.tools import control_xhand


@pytest.mark.parametrize("code, attempts", [
    (driver.COMMUNICATION_CRC_ERROR_CODE, 2),
    (driver.OVERCURRENT_WARNING_CODE, 1),
])
def test_diagnostic_requires_acknowledged_submission_before_checking_convergence(monkeypatch, code, attempts):
    cfg = HandConfig()
    hand = control_xhand.XHandDiagnostic(cfg)
    target = cfg.home_qpos_rad
    clock = SimpleNamespace(now_ns=1_000_000_000)

    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)

    mock_time = SimpleNamespace(monotonic_ns=lambda: clock.now_ns, sleep=sleep)
    monkeypatch.setattr(control_xhand, "time", mock_time)
    monkeypatch.setattr(driver, "time", mock_time)
    read = Mock(return_value=XHandState(target, None, None, clock.now_ns))
    monkeypatch.setattr(hand, "get_state", read)
    hand.control = Mock()
    hand.connected_flag = True
    hand.command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in target])
    hand.control.send_command.return_value = SimpleNamespace(
        error_code=code, error_message="SDK error",
    )

    with pytest.raises(FeedbackUnavailable, match="position submission not confirmed"):
        hand.execute_action(target)

    read.assert_called_once()
    assert hand.control.send_command.call_count == attempts
    assert hand.motion_pending

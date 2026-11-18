from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    OperationCancelled,
    RobotError,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
from dexmani_real.runtime import home


def test_home_stop_failure_reports_device_error_and_preserves_cancellation():
    robot = Mock()
    stop_failure = FeedbackUnavailable("position hold not confirmed")
    robot.stop.side_effect = [None, stop_failure]

    with pytest.raises(RobotError, match="HOME stop could not be confirmed") as error:
        home.home_robot(Mock(), robot, environment=Mock(), cancel_epoch=0, cancel_requested=lambda: True)

    assert error.value.__cause__ is stop_failure
    assert isinstance(stop_failure.__context__, OperationCancelled)
    assert str(stop_failure.__context__) == "HOME cancelled"


@pytest.mark.parametrize("write_warning", [False, True])
def test_home_waits_for_read_crc_recovery_and_requires_a_confirmed_target(monkeypatch, write_warning):
    clock = SimpleNamespace(now_ns=1_000_000_000)

    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)

    monkeypatch.setattr(home, "time", SimpleNamespace(monotonic_ns=lambda: clock.now_ns, sleep=sleep))
    cfg = replace(HandConfig(), home_timeout_s=1.0, home_max_step_deg=1.0)
    feedback = XHandFeedback()
    feedback.consume(XHandState(cfg.home_qpos_rad, None, None, clock.now_ns))
    robot = Mock(hand_config=cfg, hand_feedback=feedback)

    def service_feedback():
        robot.hand_feedback_pending = robot.service_feedback.call_count == 1
        if not robot.hand_feedback_pending:
            feedback.consume(XHandState(cfg.home_qpos_rad, None, None, clock.now_ns))
            robot.hand.feedback_overcurrent = (clock.now_ns, False, ())

    def send(*_args, **_kwargs):
        assert not robot.hand_feedback_pending
        return write_warning and robot.send_hand_home.call_count == 1

    robot.service_feedback.side_effect = service_feedback
    robot.send_hand_home.side_effect = send
    home._home_hand(robot, Mock())

    assert robot.send_hand_home.call_count == 3 + write_warning

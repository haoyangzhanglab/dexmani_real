from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.robot.xarm7.diagnostics import Trace, summarize
from scripts.tools.diagnose_xarm import read_session


@pytest.mark.parametrize('failure', [None, RuntimeError('read failed'), KeyboardInterrupt()])
def test_read_never_stops(monkeypatch, failure):
    driver = Mock()
    driver.read.return_value = (np.zeros(7), np.zeros(7))
    if failure is not None:
        driver.read.side_effect = failure
    monkeypatch.setattr('scripts.tools.diagnose_xarm.XArm7', Mock(return_value=driver))
    if failure is None:
        read_session(ArmConfig(), Trace(20), .001)
    else:
        with pytest.raises(type(failure)):
            read_session(ArmConfig(), Trace(20), .001)
    driver.close.assert_called_once()
    driver.stop.assert_not_called()
    driver.prepare_motion.assert_not_called()
    driver.move_home.assert_not_called()
    driver.servo.assert_not_called()


def test_wrapper_preserves_return_and_exception():
    trace = Trace(2)
    result = (3, [1, 2])
    assert trace.wrap('read', lambda: result)() is result
    error = RuntimeError('original failure')
    def fail():
        raise error
    with pytest.raises(RuntimeError) as caught:
        trace.wrap('read', fail)()
    assert caught.value is error
    assert trace.copy()['code'].tolist() == [3, -999]
    assert trace.wrap('read', lambda: result)() is result
    assert trace.dropped.value == 1


def test_window_frequency_includes_gaps():
    trace = Trace(10)
    for t in (100_000_000, 110_000_000, 900_000_000):
        trace.record('set_servo_angle_j', t, end=t+1)
    result = summarize(trace.copy(), [('stable', 0, 1_000_000_000)])['stable']
    assert result['effective_hz'] == 3
    assert result['over_20ms_fraction'] == .5


@pytest.mark.parametrize('mass', [0, 1.1, -1, float('nan'), float('inf')])
def test_bare_flange_mass(mass):
    cfg = replace(ArmConfig(), tcp_load_mass_kg=mass)
    if np.isfinite(mass) and mass >= 0:
        cfg.validate()
    else:
        with pytest.raises(ValueError):
            cfg.validate()


def test_spawn_trace_view_is_rebuilt():
    trace = Trace(10)
    trace.record('parent')
    assert trace.__getstate__()['_rows'] is None


def test_sdk_keyword_state_is_recorded():
    trace = Trace(10)
    trace.wrap('set_state', lambda state=0: 0)(state=4)
    assert trace.copy()['value'].tolist() == [4]


@pytest.mark.parametrize('first_mode', [0, 1])
def test_prepare_reuses_static_parameters_and_keeps_mode_enable(first_mode):
    from dexmani_real.robot.xarm7.driver import XArm7
    from unittest.mock import call
    driver = XArm7(ArmConfig())
    api = Mock()
    api.get_err_warn_code.return_value = (0, (0, 0))
    for name in ('motion_enable', 'set_mode', 'set_state', 'set_collision_sensitivity', 'set_tcp_load', 'set_joint_maxacc'):
        getattr(api, name).return_value = 0
    driver.api = api
    for mode in (first_mode, 1-first_mode):
        driver.prepare_motion(mode, check_abort=lambda: None)
    assert api.set_state.call_args_list == [call(0), call(0)]
    assert api.set_mode.call_args_list == [call(first_mode), call(1-first_mode)]
    api.set_collision_sensitivity.assert_called_once()
    api.set_tcp_load.assert_called_once()
    assert api.set_joint_maxacc.call_count == 2


@pytest.mark.parametrize('failure', [RuntimeError('startup failed'), KeyboardInterrupt()])
def test_control_failure_closes_and_interrupt_latches_emergency(monkeypatch, failure):
    from argparse import Namespace
    from scripts.tools.diagnose_xarm import control_session
    arm = Mock()
    arm.connect.side_effect = failure
    factory = Mock(return_value=arm)
    monkeypatch.setattr('scripts.tools.diagnose_xarm.ArmProcess', factory)
    with pytest.raises(type(failure)):
        control_session(ArmConfig(), Trace(10), Trace(10), Namespace(case='hold'), [])
    arm.close.assert_called_once()
    state = factory.call_args.kwargs['runtime_state']
    assert state.emergency_signal.is_set() == isinstance(failure, KeyboardInterrupt)


def test_lateness_and_submission_are_distinct():
    trace = Trace(10)
    trace.record('submit', 100, end=101, value=999)
    trace.record('tick', 110, end=110, value=105)
    trace.record('adopt', 115, end=115, value=999)
    trace.record('set_servo_angle_j', 120, end=121)
    stats = summarize(trace.copy(), [('stable', 0, 1000)])['stable']
    assert stats['tick_lateness_ms']['p50'] == 5/1e6
    assert stats['servo_lateness_ms']['p50'] == 15/1e6
    assert stats['submit_to_adopt_ms']['p50'] == 15/1e6

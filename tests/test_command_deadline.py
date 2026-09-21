"""Controlled-clock regressions at real publication and worker admission fences."""
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dexmani_real.robot.commands import ActionCandidate, publish_command
from dexmani_real.runtime.safety import (
    SafetyState, begin_motion, coupled_command_may_cross_sdk,
    invalidate_coupled_commands,
    PUBLISH_REASON_EXPIRED, PUBLISH_REASON_GENERATION, PUBLISH_REASON_FIFO_FULL,
)
import test_command_stream as transport
from test_command_stream import _publish, _GEN, _FakeSendStatus



class CommandDeadlineTest(transport._TransportTest):
    _make_state = transport.ArmWorkerConsumptionTest._make_state
    _worker = transport.WorkerEpochInterleavingTest._worker
    _setup = transport.HandWorkerConsumptionTest._setup

    def test_full_retry_keeps_target_generation_and_deadline(self):
        consumer, sdk, tick = self._worker('arm')
        now = time.monotonic_ns()
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now):
            for i in range(4):
                assert _publish(self.shared, arm=.01, expires_ns=now+1000)[1].published
            candidate, full = _publish(self.shared, arm=.02, expires_ns=now+100)
            assert full.reason == PUBLISH_REASON_FIFO_FULL
            original = candidate.arm_qpos.copy()
            for _ in range(3):
                assert publish_command(self.shared, candidate, required_safety_state=SafetyState.RUNNING).reason == PUBLISH_REASON_FIFO_FULL
            tick(_GEN)
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
            rejected = publish_command(self.shared, candidate, required_safety_state=SafetyState.RUNNING)
        assert rejected.reason == PUBLISH_REASON_EXPIRED
        assert candidate.expires_monotonic_ns == now+100
        assert candidate.run_generation == _GEN
        np.testing.assert_array_equal(candidate.arm_qpos, original)
        assert self.shared.coupled_cmd_ring.latest_sequence == 4
        assert not self.shared.error_state.value and not self.shared.estop_request.value

    def test_each_expired_head_blocks_sdk_and_the_newer_suffix(self):
        for name in ('arm', 'hand'):
            consumer, sdk, tick = self._worker(name)
            self.shared.safety_state.value = SafetyState.RUNNING
            generation = int(self.shared.run_generation.value)
            now = time.monotonic_ns()
            with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now):
                candidate, first = _publish(self.shared, generation=generation, expires_ns=now+100, **{name:.05})
                assert first.published
                assert _publish(self.shared, generation=generation, expires_ns=now+1000, **{name:.06})[1].published
            with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
                tick(generation)
                tick(generation)
            assert not (sdk.servo_calls if name=='arm' else sdk.sent)
            assert self.shared.run_generation.value == generation+1
            assert self.shared.safety_state.value == SafetyState.ARMED
            assert not self.shared.error_state.value and not self.shared.estop_request.value
            assert begin_motion(self.shared)  # Explicit new run.
            rejected = publish_command(self.shared, candidate, required_safety_state=SafetyState.RUNNING)
            assert rejected.reason == PUBLISH_REASON_GENERATION
            tick(int(self.shared.run_generation.value))
            assert not (sdk.servo_calls if name=='arm' else sdk.sent)

    def test_before_at_after_deadline_and_old_generation_priority(self):
        now = time.monotonic_ns()
        self._worker('arm')
        candidate = ActionCandidate(_GEN, now+100, arm_qpos=np.zeros(7))
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+99):
            assert publish_command(self.shared, candidate, required_safety_state=SafetyState.RUNNING).published
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
            assert not coupled_command_may_cross_sdk(self.shared, run_generation=_GEN, expires_monotonic_ns=now+100)
        new_generation = self.shared.run_generation.value
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+101):
            assert not coupled_command_may_cross_sdk(self.shared, run_generation=_GEN, expires_monotonic_ns=now+100)
        assert self.shared.run_generation.value == new_generation
        assert not self.shared.error_state.value

    def test_fast_arm_does_not_authorize_hand_after_expiry(self):
        _, arm, arm_tick = self._worker('arm')
        _, hand, hand_tick = self._worker('hand')
        now = time.monotonic_ns()
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now):
            assert _publish(self.shared, arm=.05, hand=.05, expires_ns=now+100)[1].published
            arm_tick(_GEN)
        assert len(arm.servo_calls)==1
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
            hand_tick(_GEN)
        assert hand.sent==[]
        assert self.shared.run_generation.value==_GEN+1

    def test_intermediate_slew_does_not_refresh_age_or_ack_endpoint(self):
        self._check_hand_retry_expiry(_FakeSendStatus.ACCEPTED)

    def test_crc_retry_does_not_refresh_age_or_ack_endpoint(self):
        self._check_hand_retry_expiry(_FakeSendStatus.CRC_UNCONFIRMED)

    def _check_hand_retry_expiry(self, first_status):
        self.shared.safety_state.value=SafetyState.RUNNING
        generation=int(self.shared.run_generation.value)
        _, ack, sdk, tick = self._setup([first_status, _FakeSendStatus.ACCEPTED])
        now=time.monotonic_ns()
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now):
            assert _publish(self.shared, hand=.25, generation=generation, expires_ns=now+100)[1].published
            tick()
        assert len(sdk.sent)==1
        assert ack.accepted_target_sequence==0
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
            tick()
        assert len(sdk.sent)==1
        assert ack.accepted_target_sequence==0

    def test_estop_or_new_generation_wins_over_old_expiry(self):
        generation=int(self.shared.run_generation.value)
        invalidate_coupled_commands(self.shared)
        current=int(self.shared.run_generation.value)
        assert not coupled_command_may_cross_sdk(self.shared, run_generation=generation, expires_monotonic_ns=1)
        assert self.shared.run_generation.value==current
        self.shared.estop_request.value=True
        assert not coupled_command_may_cross_sdk(self.shared, run_generation=current, expires_monotonic_ns=1)
        assert self.shared.run_generation.value==current

    def test_expired_home_never_enters_driver(self):
        from dexmani_real.robot.arm_worker import _handle_home
        driver=Mock()
        _handle_home(SimpleNamespace(arm=driver), self.shared, (np.zeros((2,7)), np.zeros(7), _GEN, 1))
        driver.home.assert_not_called()
        assert not self.shared.error_state.value

    def test_absent_actuator_cannot_skip_an_expired_head(self):
        _, sdk, tick = self._worker('arm')
        self._worker('hand')
        now = time.monotonic_ns()
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now):
            assert _publish(self.shared, hand=.05, expires_ns=now+100)[1].published
            assert _publish(self.shared, arm=.05, expires_ns=now+1000)[1].published
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', return_value=now+100):
            tick(_GEN)
            tick(_GEN)
        assert sdk.servo_calls == []
        assert self.shared.run_generation.value == _GEN+1

    def test_home_start_deadline_does_not_expire_an_admitted_long_trajectory(self):
        from dexmani_real.robot.arm_worker import _handle_home
        self.shared.safety_state.value = SafetyState.ARMED
        now = [time.monotonic_ns()]
        self.shared.quit_requested = SimpleNamespace(value=False)
        self.shared.arm_home_completed_generation = SimpleNamespace(value=0)
        self.shared.set_heartbeat = lambda *args: None
        driver = Mock()
        def home(*args, abort_check, **kwargs):
            now[0] += 10_000_000_000
            assert abort_check() is None
        driver.home.side_effect = home
        with patch('dexmani_real.runtime.safety.time.monotonic_ns', side_effect=lambda: now[0]):
            _handle_home(SimpleNamespace(arm=driver), self.shared,
                         (np.zeros((2,7)), np.zeros(7), _GEN, now[0]+100))
        driver.home.assert_called_once()
        assert self.shared.arm_home_completed_generation.value == _GEN


@pytest.mark.parametrize('value', [None, 0, -1, float('nan'), float('inf'), True])
def test_live_budget_rejected_before_device_startup(value):
    from dexmani_real.config.experiment import resolve_experiment_config
    from dexmani_real.teleop.session import run_teleop_experiment
    runtime=resolve_experiment_config()
    runtime=replace(runtime, safety=replace(runtime.safety, max_dispatch_delay_s=value))
    with patch('dexmani_real.teleop.session.RuntimeChannels.create', side_effect=AssertionError('hardware startup reached')):
        with pytest.raises(ValueError):
            run_teleop_experiment(runtime)

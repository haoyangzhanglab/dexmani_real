"""Offline tests for recorder fault isolation and run-owner trial counting (T5).

Covers the acceptance matrix:
* V15 START refusal / in-band recorder failures change only evidence state —
  the RUNNING trial and motion are never terminated by recording, and a
  refused START never rejects a valid B;
* V16 trials and saved episodes are independent counts, a trial counts
  exactly once, directory names follow the trial sequence (a failed save
  never reuses a name), and an evidence-role service failure during RUNNING
  remains evidence-only across subsequent ARMED and RUNNING trials.

The runner is exercised through its real methods against a fake recorder and
a fake shared-state object wired to the real safety primitives; no hardware,
SDK, or spawned process is involved except the supervisor thread tests, which
use the real ``run_supervisor`` loop over the same fakes.
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace

# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    from dexmani_real.deployment.runner import PolicyRunner
    from dexmani_real.recording.client import RecorderStopResult
    from dexmani_real.runtime.safety import SafetyState
    from dexmani_real.runtime.supervisor import run_supervisor
    from dexmani_real.utils.log import ThrottledWarner

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    _IMPORT_ERROR = exc


class _FakeRecorderClient:
    def __init__(
        self,
        *,
        start_ok: bool = True,
        transport_unavailable: bool = False,
        last_error: str | None = None,
    ) -> None:
        self.start_calls: list[str | None] = []
        self.stop_calls: list[tuple[bool, str]] = []
        self._start_ok = start_ok
        self.transport_unavailable = transport_unavailable
        self.last_error = last_error
        self.is_recording = False
        self.stop_pending = False
        self._result: RecorderStopResult | None = None

    def start_episode(self, *, task_label="", operator="", episode_name=None):
        self.start_calls.append(episode_name)
        if self._start_ok:
            self.is_recording = True
        return self._start_ok

    def stop_episode(self, save=True, reason="", *, retain_partial=False):
        self.stop_calls.append((bool(save), reason))
        self.is_recording = False
        self.stop_pending = False
        return None

    def poll_stop(self) -> "RecorderStopResult":
        result = self._result
        self._result = None
        return result or RecorderStopResult(done=False)

    def join_stop(self, timeout=None):
        return self.poll_stop()


def _fake_shared(*, safety_state: int = int(SafetyState.ARMED)):
    return SimpleNamespace(
        motion_lock=threading.RLock(),
        start_request=SimpleNamespace(value=False),
        stop_request=SimpleNamespace(value=0),
        physical_home_completed=SimpleNamespace(value=True),
        is_recording=SimpleNamespace(value=False),
        quit_requested=SimpleNamespace(value=False),
        session_failed=SimpleNamespace(value=False),
        evidence_failed=SimpleNamespace(value=False),
        error_state=SimpleNamespace(value=False),
        estop_request=SimpleNamespace(value=False),
        is_running=SimpleNamespace(value=True),
        safety_state=SimpleNamespace(value=safety_state),
        run_generation=SimpleNamespace(value=5),
        run_generation_base_sequence=SimpleNamespace(value=0),
        run_started_monotonic_ns=SimpleNamespace(value=0),
        run_started_generation=SimpleNamespace(value=0),
        run_ended_generation=SimpleNamespace(value=0),
        run_ended_monotonic_ns=SimpleNamespace(value=0),
        run_ended_reason=SimpleNamespace(value=0),

        arm_cmd_consumed_sequence=SimpleNamespace(value=-1),
        hand_cmd_consumed_sequence=SimpleNamespace(value=-1),
        coupled_cmd_ring=SimpleNamespace(latest_sequence=0, maxlen=4),
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class _RunnerTest(unittest.TestCase):
    def _runner(
        self,
        *,
        shared=None,
        recorder=None,
        num_trials: int = 2,
        run_started_ns: int | None = None,
    ) -> "PolicyRunner":
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.shared = shared if shared is not None else _fake_shared()
        runner.recorder = recorder
        runner.recording_config = (
            SimpleNamespace(task_label="t", operator="o")
            if recorder is not None
            else None
        )
        runner.num_trials = num_trials
        runner.completed_trials = 0
        runner.saved_episodes = 0
        runner.recording_unavailable = False
        runner.evidence_failed = False
        runner.evidence_failure_reason = None
        runner._evidence_logged_this_trial = False
        runner._evidence_warn = ThrottledWarner(interval_s=2.0)
        runner._pending_stop_reason = None
        runner._recording_trial_id = 1 if recorder is not None and run_started_ns is not None else None
        runner._recording_outcome_consumed = runner._recording_trial_id is None
        runner._recorder_start_wait_ms = 0.0
        runner.run_started_ns = run_started_ns
        if run_started_ns is not None:
            runner.shared.run_started_monotonic_ns.value = run_started_ns
            runner.shared.safety_state.value = int(SafetyState.RUNNING)
        runner.run_generation = None
        runner._trial_generation = runner.shared.run_started_generation.value
        runner.last_publication_ns = None
        runner.previous_arm_command_qpos = None
        runner._pending_dispatch = None
        runner.actions = deque()
        runner.execute = False
        runner.max_running_s = 60.0
        runner.session_publication_count = 0
        runner.session_running_ns = 0
        runner.session_inference_ms = []
        runner.step_dt_ns = 62_500_000
        runner.next_record_ns = 0
        runner.runtime = SimpleNamespace(policy=SimpleNamespace())
        runner.model_runtime = SimpleNamespace(reset_episode=lambda: None)
        return runner


class TrialCountingTest(_RunnerTest):
    def test_trial_counts_once_and_ends_the_run_at_budget(self):
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        shared.run_started_monotonic_ns.value = time.monotonic_ns()
        runner = self._runner(shared=shared, run_started_ns=1)
        runner._finish_episode("operator stop", stop_reason="operator", aborted=False)
        self.assertEqual(runner.completed_trials, 1)
        self.assertFalse(bool(shared.quit_requested.value))
        # A second finish without a new epoch must not double-count.
        runner._finish_episode("ignored")
        self.assertEqual(runner.completed_trials, 1)
        # The second begun trial ends the run at the trial budget.
        runner.run_started_ns = 2
        shared.run_started_monotonic_ns.value = 2
        shared.safety_state.value = int(SafetyState.RUNNING)
        runner._finish_episode("operator stop", stop_reason="operator", aborted=False)
        self.assertEqual(runner.completed_trials, 2)
        self.assertTrue(bool(shared.quit_requested.value))

    def test_estop_ended_trial_counts_and_keeps_its_running_time(self):
        """V20: an epoch that truly began counts even without a recorder.

        Without this the statistics denominator would lose that trial's
        RUNNING wall time while its publications stayed in the numerator.
        """
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        shared.estop_request.value = True
        runner = self._runner(
            shared=shared, recorder=None, run_started_ns=time.monotonic_ns()
        )
        runner._handle_run_boundary()
        self.assertEqual(runner.completed_trials, 1)
        self.assertGreater(runner.session_running_ns, 0)
        self.assertIsNone(runner.run_started_ns)
        self.assertEqual(int(shared.safety_state.value), int(SafetyState.FAULT))

    def test_saved_count_is_independent_of_trial_count(self):
        recorder = _FakeRecorderClient()
        runner = self._runner(recorder=recorder, run_started_ns=1)
        runner._finish_episode("stop", stop_reason="operator", aborted=False)
        runner._poll_recorder()
        self.assertEqual(runner.completed_trials, 1)
        self.assertEqual(runner.saved_episodes, 0)
        recorder._result = RecorderStopResult(
            done=True, saved=True, path="p", frame_count=5, reason="operator"
        )
        runner._complete_recording(recorder.poll_stop())
        self.assertEqual(runner.saved_episodes, 1)
        self.assertEqual(runner.completed_trials, 1)


class RecordingOutcomeOwnershipTest(_RunnerTest):
    """A trial owns at most one terminal recording outcome."""

    def test_trial_without_recording_leaves_no_pending_outcome(self):
        recorder = _FakeRecorderClient(start_ok=False, last_error="refused")
        runner = self._runner(recorder=recorder, run_started_ns=1)
        recorder.is_recording = False
        recorder.stop_pending = False
        runner._recording_trial_id = None
        runner._recording_outcome_consumed = True
        runner._finish_episode("stop", stop_reason="operator", aborted=False)
        # Nothing was recorded, so the shutdown path must not later consume an
        # older verdict for this trial (double-counted saves / wrong trial id).
        self.assertIsNone(runner._pending_stop_reason)
        self.assertTrue(runner._recording_outcome_consumed)
        self.assertEqual(recorder.stop_calls, [])

    def test_recording_trial_marks_then_consumes_its_outcome(self):
        recorder = _FakeRecorderClient()
        recorder.is_recording = True
        runner = self._runner(recorder=recorder, run_started_ns=1)
        runner._finish_episode("stop", stop_reason="operator", aborted=False)
        self.assertEqual(runner._pending_stop_reason, "operator")
        self.assertFalse(runner._recording_outcome_consumed)
        self.assertEqual(recorder.stop_calls, [(True, "operator")])

        recorder._result = RecorderStopResult(
            done=True, saved=True, path="p", frame_count=9, reason="operator"
        )
        runner._poll_recorder()  # poll path consumes the terminal verdict
        self.assertEqual(runner.saved_episodes, 1)
        self.assertTrue(runner._recording_outcome_consumed)
        self.assertIsNone(runner._pending_stop_reason)


class EvidenceIsolationTest(_RunnerTest):
    def test_recorder_failure_does_not_end_control(self):
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        recorder = _FakeRecorderClient()
        runner = self._runner(shared=shared, recorder=recorder, run_started_ns=1)
        recorder._result = RecorderStopResult(
            done=True, saved=False, error="writer died", reason="camera_writer_error"
        )
        keeps_running = runner._poll_recorder()
        self.assertTrue(keeps_running)
        self.assertEqual(runner.run_started_ns, 1)  # trial continues
        self.assertTrue(runner.evidence_failed)
        self.assertIn("writer died", runner.evidence_failure_reason)
        self.assertFalse(bool(shared.quit_requested.value))
        self.assertFalse(bool(shared.session_failed.value))
        self.assertFalse(bool(shared.error_state.value))

    def test_unexpected_finalization_is_evidence_only(self):
        recorder = _FakeRecorderClient()
        runner = self._runner(recorder=recorder, run_started_ns=1)
        recorder._result = RecorderStopResult(
            done=True, saved=False, reason="recorder_process_shutdown", frame_count=3
        )
        self.assertTrue(runner._poll_recorder())
        self.assertTrue(runner.evidence_failed)
        self.assertEqual(runner.run_started_ns, 1)

    def test_refused_start_does_not_reject_valid_b(self):
        shared = _fake_shared(safety_state=int(SafetyState.ARMED))
        recorder = _FakeRecorderClient(
            start_ok=False, last_error="EpisodeRecorder refused start"
        )
        runner = self._runner(shared=shared, recorder=recorder)
        shared.start_request.value = True
        runner._start_requested_episode()
        # The trial began WITHOUT recording: B was not rejected by evidence.
        self.assertIsNotNone(runner.run_started_ns)
        self.assertEqual(int(shared.safety_state.value), int(SafetyState.RUNNING))
        self.assertTrue(runner.evidence_failed)
        self.assertFalse(runner.recording_unavailable)
        self.assertEqual(recorder.start_calls, ["episode_001"])

    def test_transport_dead_start_disables_session_recording(self):
        shared = _fake_shared(safety_state=int(SafetyState.ARMED))
        recorder = _FakeRecorderClient(
            start_ok=False, transport_unavailable=True, last_error="channel corrupt"
        )
        runner = self._runner(shared=shared, recorder=recorder)
        shared.start_request.value = True
        runner._start_requested_episode()
        self.assertIsNotNone(runner.run_started_ns)
        self.assertTrue(runner.recording_unavailable)
        # A later trial never reuses the unfinished channel: no second START.
        runner._finish_episode("stop", stop_reason="operator", aborted=False)
        shared.start_request.value = True
        shared.safety_state.value = int(SafetyState.ARMED)
        runner.run_started_ns = None
        runner._start_requested_episode()
        self.assertEqual(recorder.start_calls, ["episode_001"])
        self.assertIsNotNone(runner.run_started_ns)

    def test_evidence_failure_keeps_next_trial_unrecorded(self):
        shared = _fake_shared(safety_state=int(SafetyState.ARMED))
        recorder = _FakeRecorderClient(start_ok=False, last_error="nope")
        runner = self._runner(shared=shared, recorder=recorder)
        shared.start_request.value = True
        runner._start_requested_episode()  # trial 1: recording refused
        runner._finish_episode("stop", stop_reason="operator", aborted=False)
        # Trial 1 saved nothing; its name must not be reused.
        shared.start_request.value = True
        shared.safety_state.value = int(SafetyState.ARMED)
        runner.run_started_ns = None
        runner._start_requested_episode()
        self.assertEqual(recorder.start_calls, ["episode_001"])
        self.assertIsNotNone(runner.run_started_ns)  # failed evidence never gates trial 2


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class SessionStatisticsTest(_RunnerTest):
    """V20: the four session statistics use real sources and denominators."""

    def _stats_runner(self):
        runner = self._runner(num_trials=2)
        runner.control_period_s = 0.0625
        return runner

    def test_effective_hz_and_percentiles_from_real_accumulators(self):
        runner = self._stats_runner()
        runner.session_publication_count = 100
        runner.session_running_ns = 5_000_000_000  # 5 s of RUNNING wall time
        runner.session_inference_ms = [10.0, 20.0, 30.0, 40.0]
        with self.assertLogs("dexmani_real.deployment.runner", level="INFO") as logs:
            runner._log_session_statistics()
        line = "\n".join(logs.output)
        self.assertIn("nominal_hz=16.000", line)  # 1/0.0625
        self.assertIn("effective_publication_hz=20.000", line)  # 100 commits / 5 s
        self.assertIn("publications=100", line)
        self.assertIn("predict_samples=4", line)
        self.assertIn("inference_ms_mean=25.000", line)
        self.assertIn("inference_ms_p95=38.500", line)

    def test_unavailable_markers_without_running_time_or_samples(self):
        runner = self._stats_runner()
        with self.assertLogs("dexmani_real.deployment.runner", level="INFO") as logs:
            runner._log_session_statistics()
        line = "\n".join(logs.output)
        self.assertIn("effective_publication_hz=unavailable", line)
        self.assertIn("inference_ms_mean=unavailable", line)
        self.assertIn("inference_ms_p95=unavailable", line)
        self.assertIn("predict_samples=0", line)


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class TerminalResultConsumptionTest(unittest.TestCase):
    """V16: a terminal recorder verdict is consumed exactly once."""

    def test_done_result_is_delivered_once_after_transport_loss(self):
        from queue import Empty

        from dexmani_real.recording.client import (
            RecorderClient,
            RecorderStopResult,
        )

        class _EmptyQueue:
            def get_nowait(self):
                raise Empty

        class _Shared:
            record_result_q = _EmptyQueue()
            is_ready = staticmethod(lambda name: True)
            is_running = SimpleNamespace(value=True)
            evidence_failed = SimpleNamespace(value=False)
            record_sample_ring = SimpleNamespace(latest_sequence=0, maxlen=4)
            recorder_consumed_sequence = SimpleNamespace(value=0)
            set_heartbeat = staticmethod(lambda *a: None)

        client = RecorderClient(_Shared())
        client._unavailable = True
        client._last_stop_result = RecorderStopResult(
            done=True, saved=True, path="p", frame_count=7, reason="manual"
        )
        first = client.poll_stop()
        second = client.poll_stop()
        self.assertTrue(first.done)
        self.assertTrue(first.saved)
        self.assertFalse(second.done)
        # join_stop (the shutdown path) must not re-deliver the consumed verdict
        # either: that would let the owner re-run its completion bookkeeping.
        joined = client.join_stop()
        self.assertTrue(joined.done)
        self.assertFalse(joined.saved)
        # The consumer's evidence counter therefore increments exactly once.
        from dexmani_real.recording.client import RecordingFinished

        client._last_stop_result = None
        client._terminal_result_delivered = False
        client._unavailable = False
        result = client._finish(
            RecordingFinished(saved=True, path="p", frame_count=7, reason="manual")
        )
        self.assertTrue(result.done)
        self.assertTrue(client._terminal_result_delivered)


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class SupervisorEvidenceDeferralTest(unittest.TestCase):
    """An evidence-role service failure never terminates a RUNNING trial."""

    def _run_supervisor_thread(self, shared, **kwargs):
        result: dict = {}

        def target():
            result["outcome"] = run_supervisor(
                shared,
                [],
                status_interval_s=3600.0,
                heartbeat_timeouts_s={},
                supervisor_hz=500.0,
                service_process_names=("recorder",),
                **kwargs,
            )

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, result

    def test_service_failure_survives_armed_and_next_trial_until_q(self):
        from unittest.mock import patch
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        shared.evidence_failed.value = True
        states = iter((SafetyState.ARMED, SafetyState.RUNNING, SafetyState.ARMED))
        visited = []
        def advance(_delay):
            state = next(states)
            visited.append(state)
            shared.safety_state.value = int(state)
            if len(visited) == 3:
                shared.quit_requested.value = True
        dead = SimpleNamespace(name="recorder", exitcode=1)
        with patch("dexmani_real.runtime.supervisor.time.sleep", side_effect=advance):
            reason, normal = run_supervisor(shared, [dead], heartbeat_timeouts_s={},
                service_process_names={"recorder"})
        self.assertEqual(len(visited), 3)
        self.assertEqual(reason, "shutdown requested")
        self.assertTrue(normal)
        self.assertFalse(shared.session_failed.value)

    def test_estop_still_wins_immediately(self):
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        shared.session_failed.value = True
        shared.estop_request.value = True
        thread, result = self._run_supervisor_thread(shared)
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        exit_reason, normal_exit = result["outcome"]
        self.assertEqual(exit_reason, "e-stop requested")
        self.assertFalse(normal_exit)
        self.assertEqual(int(shared.safety_state.value), int(SafetyState.FAULT))


if __name__ == "__main__":
    unittest.main()

class RecordingAcrossTrialsTest(_RunnerTest):
    def test_capacity_saves_once_without_ending_trial(self):
        recorder = _FakeRecorderClient()
        runner = self._runner(recorder=recorder, run_started_ns=1)
        runner._recording_trial_id = 1
        runner._recording_outcome_consumed = False
        recorder._result = RecorderStopResult(done=True, saved=True, reason="max_frames", frame_count=2)
        self.assertTrue(runner._poll_recorder())
        self.assertEqual(runner.run_started_ns, 1)
        self.assertEqual(runner.completed_trials, 0)
        self.assertEqual(runner.saved_episodes, 1)
        runner._poll_recorder()
        self.assertEqual(runner.saved_episodes, 1)

    def test_pending_old_result_keeps_identity_through_unrecorded_trial(self):
        from unittest.mock import patch
        recorder = _FakeRecorderClient()
        recorder.stop_pending = True
        runner = self._runner(recorder=recorder, num_trials=3)
        runner.completed_trials = 1
        runner._recording_trial_id = 1
        runner._recording_outcome_consumed = False
        runner._pending_stop_reason = "operator"
        runner.shared.is_recording.value = True
        runner.shared.start_request.value = True
        runner._start_requested_episode()
        self.assertIsNotNone(runner.run_started_ns)
        self.assertEqual(recorder.start_calls, [])
        self.assertTrue(runner.shared.is_recording.value)
        runner._finish_episode("timeout", stop_reason="timeout")
        self.assertEqual(runner._pending_stop_reason, "operator")
        self.assertFalse(runner._recording_outcome_consumed)
        recorder._result = RecorderStopResult(done=True, saved=True, reason="operator", frame_count=3)
        with patch("builtins.print") as output:
            runner._poll_recorder()
        self.assertIn("Trial 1/3", str(output.call_args_list))
        self.assertEqual(runner.saved_episodes, 1)
        self.assertEqual(runner.completed_trials, 2)

    def test_all_start_outcomes_recheck_stop_epoch_and_physical_state(self):
        from unittest.mock import patch
        from dexmani_real.runtime.safety import request_policy_stop
        for started in (False, True):
            for change in ("generation", "S", "Q", "physical"):
                with self.subTest(started=started, change=change):
                    recorder = _FakeRecorderClient(start_ok=started)
                    runner = self._runner(recorder=recorder)
                    runner.shared.start_request.value = True
                    original = recorder.start_episode
                    def prepare(**kwargs):
                        if change == "generation":
                            runner.shared.run_generation.value += 1
                        elif change == "S":
                            request_policy_stop(runner.shared)
                        elif change == "Q":
                            runner.shared.quit_requested.value = True
                        return original(**kwargs)
                    recorder.start_episode = prepare
                    with patch("dexmani_real.deployment.runner._physical_start_pose_rejection",
                               side_effect=[None, "moved" if change == "physical" else None]):
                        runner._start_requested_episode()
                    self.assertIsNone(runner.run_started_ns)
                    self.assertEqual(runner.completed_trials, 0)

    def test_evidence_and_quit_priority(self):
        from dexmani_real.runtime.supervisor import supervisor_exit_reason
        from dexmani_real.runtime.status import ExitReason
        shared = _fake_shared()
        shared.quit_requested.value = True
        dead = SimpleNamespace(name="recorder", exitcode=1)
        self.assertEqual(supervisor_exit_reason(shared, [dead], {}, {},
            service_process_names={"recorder"}), ExitReason.EXPLICIT_QUIT)

class EvidenceStartupTest(unittest.TestCase):
    def test_optional_failure_keeps_handles_and_required_failure_is_terminal(self):
        from dexmani_real.runtime.supervisor import start_evidence_services
        for critical_alive in (True, False):
            shared = _fake_shared()
            shared.is_ready = lambda name: False
            critical = SimpleNamespace(name="arm", is_alive=lambda: critical_alive)
            optional = SimpleNamespace(name="recorder", pid=None, is_alive=lambda: False)
            def start():
                optional.pid = 123
            optional.start = start
            started = [critical]
            if critical_alive:
                self.assertFalse(start_evidence_services(shared, [optional], {"recorder": 1.},
                    critical_processes=[critical], started_processes=started))
            else:
                with self.assertRaisesRegex(RuntimeError, "critical startup"):
                    start_evidence_services(shared, [optional], {"recorder": 1.},
                        critical_processes=[critical], started_processes=started)
            self.assertIn(optional, started)
            self.assertTrue(shared.evidence_failed.value)
            self.assertFalse(shared.quit_requested.value)

class SessionResultFactsTest(unittest.TestCase):
    def test_evidence_policy_and_cleanup_are_independent(self):
        from dexmani_real.deployment.session import _session_result_facts
        from dexmani_real.runtime.processes import ShutdownReport, ProcessExit
        for evidence, policy, closed, expected_record, expected_cleanup in (
            (True, False, True, "failed", "clean"),
            (False, True, True, "no evidence failure observed", "clean"),
            (False, False, False, "no evidence failure observed", "incomplete-or-failed"),
        ):
            shared = _fake_shared(safety_state=int(SafetyState.DISARMED))
            shared.evidence_failed.value, shared.session_failed.value = evidence, policy
            report = ShutdownReport((ProcessExit("recorder", 0, "graceful"),), closed)
            self.assertEqual(_session_result_facts(shared, report, recording_enabled=True,
                normal_exit=True), (expected_record, expected_cleanup, False))

class VerifiedEvidenceCleanupTest(unittest.TestCase):
    def test_unconfirmed_process_never_releases_ipc(self):
        from dexmani_real.runtime.processes import shutdown_processes_verified
        from unittest.mock import Mock
        shared = _fake_shared()
        shared.close = Mock(return_value=True)
        process = SimpleNamespace(name="recorder", exitcode=None, is_alive=lambda: True,
            join=lambda **kw: None, terminate=lambda: None, kill=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "confirmed stopped"):
            shutdown_processes_verified(shared, [process], graceful_timeout_s=0,
                terminate_timeout_s=0, kill_timeout_s=0, service_process_names={"recorder"})
        shared.close.assert_not_called()

    def test_confirmed_abnormal_evidence_exit_can_release_resources(self):
        from dexmani_real.runtime.processes import shutdown_processes_verified
        shared = _fake_shared()
        shared.close = lambda: True
        process = SimpleNamespace(name="recorder", exitcode=1, is_alive=lambda: False,
            join=lambda **kw: None)
        report = shutdown_processes_verified(shared, [process], graceful_timeout_s=0,
            service_process_names={"recorder"}, disarm_if_clean=True)
        self.assertTrue(report.shared_closed)
        self.assertTrue(shared.evidence_failed.value)
        self.assertFalse(shared.error_state.value)
        self.assertFalse(shared.session_failed.value)

class RunTerminationFactsTest(_RunnerTest):
    def test_external_stop_before_late_predict_return_uses_actual_end(self):
        from unittest.mock import patch
        from test_sync_policy_timing import _Clock
        from dexmani_real.runtime.safety import request_policy_stop
        clock = _Clock(1_000_000_000)
        runner = self._runner()
        with patch("dexmani_real.runtime.safety.time", clock), patch("dexmani_real.deployment.runner.time", clock):
            runner.shared.start_request.value = True
            runner._start_requested_episode()
            clock.ns = 3_000_000_000  # relative t=2: external S during predict
            request_policy_stop(runner.shared)
            clock.ns = 11_000_000_000  # relative t=10: predict finally returns
            runner._handle_run_boundary()
        self.assertEqual(runner.session_running_ns, 2_000_000_000)
        self.assertEqual(runner.completed_trials, 1)

    def test_epoch_clear_reports_the_entire_unpublished_suffix_once(self):
        runner = self._runner(run_started_ns=1)
        runner.run_generation = 5
        runner.actions = deque(range(7))  # first of eight already committed
        runner._pending_dispatch = object()
        with self.assertLogs(level="INFO") as logs:
            runner._clear_execution(None)
        drops = [line for line in logs.output if "[DROP]" in line]
        self.assertEqual(len(drops), 1)
        self.assertIn("remaining=7", drops[0])

class FirstTerminalFactTest(_RunnerTest):
    def test_first_fact_survives_repeated_stop_home_cleanup_and_next_trial(self):
        from unittest.mock import patch
        from test_sync_policy_timing import _Clock
        from dexmani_real.runtime.safety import (RunEndReason, request_policy_stop,
            read_run_end, revoke_motion, invalidate_coupled_commands)
        clock = _Clock(0)
        runner = self._runner(num_trials=3)
        with patch("dexmani_real.runtime.safety.time", clock), patch("dexmani_real.deployment.runner.time", clock):
            runner.shared.start_request.value = True
            runner._start_requested_episode()
            generation = runner.run_generation
            clock.ns = 2_000_000_000
            request_policy_stop(runner.shared)
            first = read_run_end(runner.shared)
            self.assertEqual(first[0], generation)
            self.assertEqual(first[2], RunEndReason.OPERATOR)
            clock.ns = 10_000_000_000
            request_policy_stop(runner.shared)
            revoke_motion(runner.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            invalidate_coupled_commands(runner.shared)  # ARMED/home rebase
            self.assertEqual(read_run_end(runner.shared)[1], first[1])
            runner._handle_run_boundary()
            self.assertEqual(runner.session_running_ns, 2_000_000_000)
            runner.shared.start_request.value = True
            runner._start_requested_episode()
            invalidate_coupled_commands(runner.shared)  # RUNNING teleop-style pause
            self.assertEqual(read_run_end(runner.shared)[0], generation)
            clock.ns = 13_000_000_000
            request_policy_stop(runner.shared)
            runner._handle_run_boundary()
            self.assertEqual(runner.session_running_ns, 5_000_000_000)
            self.assertEqual(runner.completed_trials, 2)

    def test_parent_budget_uses_actual_revoke_time_and_timeout_reason(self):
        from unittest.mock import patch
        from test_sync_policy_timing import _Clock
        from dexmani_real.runtime.safety import RunEndReason, read_run_end
        clock = _Clock(1_000_000_000)
        runner = self._runner()
        with patch("dexmani_real.runtime.safety.time", clock), patch("dexmani_real.deployment.runner.time", clock):
            runner.shared.start_request.value = True
            runner._start_requested_episode()
            clock.ns = 3_100_000_000  # parent's poll actually revokes 0.1s after budget
            def next_poll(_seconds):
                runner.shared.quit_requested.value = True
            with patch("dexmani_real.runtime.supervisor.time.monotonic_ns", clock.monotonic_ns), patch("dexmani_real.runtime.supervisor.time.sleep", side_effect=next_poll):
                run_supervisor(runner.shared, [], heartbeat_timeouts_s={}, max_running_s=2.)
            self.assertEqual(read_run_end(runner.shared)[2], RunEndReason.TIMEOUT)
            clock.ns = 11_000_000_000
            runner._handle_run_boundary()
            self.assertEqual(runner.session_running_ns, 2_100_000_000)

class BlockingPredictionStopTest(_RunnerTest):
    def test_real_active_tick_prediction_late_return_has_two_second_denominator(self):
        import numpy as np
        from unittest.mock import patch
        from test_sync_policy_timing import SyncPolicyTimingTest
        from dexmani_real.runtime.safety import request_policy_stop
        helper = SyncPolicyTimingTest()
        helper.setUp()
        helper._install_observation_fakes()
        self.addCleanup(helper.doCleanups)
        runner = self._runner()
        runner.max_running_ns = None
        runner.policy_spec = SimpleNamespace(n_obs_steps=2, n_action_steps=8, control_action_dim=19)
        runner.fingertip_runtime = None
        def predict(observation):
            helper.clock.ns = 2_000_000_000
            request_policy_stop(runner.shared)
            helper.clock.ns = 10_000_000_000
            return np.zeros((8, 19))
        runner.model_runtime.predict = predict
        with patch("dexmani_real.runtime.safety.time", helper.clock):
            runner.shared.start_request.value = True
            runner._start_requested_episode()
            with self.assertLogs("dexmani_real.deployment.runner", level="WARNING") as logs:
                runner._run_active_tick(0)
            self.assertIn("remaining=8", "\n".join(logs.output))
            runner._handle_run_boundary()
        self.assertEqual(runner.session_running_ns, 2_000_000_000)
        self.assertEqual(runner.session_inference_ms, [10000.])
        self.assertEqual(runner.session_publication_count, 0)

class RecorderExceptionBoundaryTest(_RunnerTest):
    def test_stop_exception_cannot_interrupt_trial_bookkeeping(self):
        recorder = _FakeRecorderClient()
        recorder.is_recording = True
        runner = self._runner(recorder=recorder, run_started_ns=1)
        def stop(**kwargs):
            raise OSError("broken STOP channel")
        recorder.stop_episode = stop
        runner._finish_episode("operator", stop_reason="operator")
        self.assertEqual(runner.completed_trials, 1)
        self.assertIsNone(runner.run_started_ns)
        self.assertTrue(runner.shared.evidence_failed.value)
        self.assertFalse(runner.shared.error_state.value)

class CleanupSummaryOutputTest(unittest.TestCase):
    def test_unverified_cleanup_is_visible_without_relabeling_policy_as_recording_failure(self):
        from unittest.mock import patch
        from dexmani_real.deployment.session import _report_session_end
        shared = _fake_shared()
        shared.session_failed.value = True
        with patch("builtins.print") as output:
            self.assertFalse(_report_session_end(shared, None, recording_enabled=True,
                normal_exit=False, exit_reason="policy failure"))
        text = str(output.call_args_list)
        self.assertIn("cleanup_status  = incomplete-or-failed", text)
        self.assertIn("recording_status= no evidence failure observed", text)

class LateRecordingDuringTrialTest(_RunnerTest):
    def test_old_result_during_new_trial_does_not_stop_or_record_new_trial(self):
        from unittest.mock import patch
        recorder = _FakeRecorderClient()
        recorder.stop_pending = True
        runner = self._runner(recorder=recorder, num_trials=3)
        runner.completed_trials = 1
        runner._recording_trial_id = 1
        runner._recording_outcome_consumed = False
        runner._pending_stop_reason = "operator"
        runner.shared.start_request.value = True
        runner.shared.is_recording.value = True
        runner._start_requested_episode()
        start = runner.run_started_ns
        recorder._result = RecorderStopResult(done=True, saved=True, frame_count=2, reason="operator")
        with patch("builtins.print") as output:
            runner._poll_recorder()
        self.assertIn("Trial 1/3", str(output.call_args_list))
        self.assertEqual(runner.completed_trials, 1)
        self.assertEqual(runner.run_started_ns, start)
        self.assertEqual(runner.saved_episodes, 1)
        runner._finish_episode("timeout", stop_reason="timeout")
        self.assertEqual(runner.saved_episodes, 1)
        self.assertEqual(recorder.stop_calls, [])

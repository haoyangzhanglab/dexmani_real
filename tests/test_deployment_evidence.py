"""Offline tests for recorder fault isolation and run-owner trial counting (T5).

Covers the acceptance matrix:
* V15 START refusal / in-band recorder failures change only evidence state —
  the RUNNING trial and motion are never terminated by recording, and a
  refused START never rejects a valid B;
* V16 trials and saved episodes are independent counts, a trial counts
  exactly once, directory names follow the trial sequence (a failed save
  never reuses a name), and an evidence-role service failure during RUNNING
  defers to the natural end of control instead of terminating it.

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
    from dexmani_real.deployment.executor import PolicyRunner
    from dexmani_real.deployment.metrics import PolicyStats
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

    def stop_episode(self, save=True, reason=""):
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
        error_state=SimpleNamespace(value=False),
        estop_request=SimpleNamespace(value=False),
        is_running=SimpleNamespace(value=True),
        safety_state=SimpleNamespace(value=safety_state),
        run_generation=SimpleNamespace(value=5),
        run_generation_base_sequence=SimpleNamespace(value=0),
        run_started_monotonic_ns=SimpleNamespace(value=0),
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
        runner._recording_outcome_consumed = True
        runner._recorder_start_wait_ms = 0.0
        runner.run_started_ns = run_started_ns
        runner.run_generation = None
        runner.last_publication_ns = None
        runner.previous_arm_command_qpos = None
        runner._observation_waiting_since_ns = None
        runner._pending_dispatch = None
        runner.actions = deque()
        runner.chunk_sources = {}
        runner.chunk_action_index = 0
        runner.observation_id = 0
        runner.stats = PolicyStats()
        runner.last_metrics_flush_ns = 0
        runner.execute = False
        runner.max_running_s = 60.0
        runner.session_publication_count = 0
        runner.session_running_ns = 0
        runner.session_inference_ms = []
        runner.step_dt_ns = 62_500_000
        runner.next_record_ns = 0
        runner.runtime = SimpleNamespace(policy=SimpleNamespace())
        runner.model_runtime = SimpleNamespace(reset_episode=lambda: None)
        runner._fifo_wait = SimpleNamespace(
            waiting=False,
            note_full=lambda *a: None,
            note_committed=lambda: None,
            note_dropped=lambda reason: None,
        )
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
        recorder._result = RecorderStopResult(
            done=True, saved=False, error=None, path=None, frame_count=0, reason="d"
        )
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

    def test_directory_names_follow_trial_sequence_not_saves(self):
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
        self.assertEqual(recorder.start_calls, ["episode_001", "episode_002"])


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
        with self.assertLogs("dexmani_real.deployment.executor", level="INFO") as logs:
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
        with self.assertLogs("dexmani_real.deployment.executor", level="INFO") as logs:
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

    def test_service_failure_defers_until_control_ends_naturally(self):
        shared = _fake_shared(safety_state=int(SafetyState.RUNNING))
        shared.run_started_monotonic_ns.value = time.monotonic_ns()
        shared.session_failed.value = True  # evidence latch during RUNNING
        thread, result = self._run_supervisor_thread(shared)
        try:
            time.sleep(0.2)
            self.assertTrue(thread.is_alive(), "supervisor must not exit during RUNNING")
            # Control ends naturally: motion fenced, operator quit observed.
            shared.safety_state.value = int(SafetyState.ARMED)
            shared.quit_requested.value = True
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            exit_reason, normal_exit = result["outcome"]
            self.assertEqual(exit_reason, "session/service failure")
            self.assertTrue(normal_exit)
            self.assertTrue(bool(shared.session_failed.value))
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)

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

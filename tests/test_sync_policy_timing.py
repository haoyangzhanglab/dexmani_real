"""Offline regression tests for synchronous policy timing semantics.

These exercise the real ``PolicyRunner._run_active_tick`` scheduler against a
fake clock, fake model, and monkeypatched observation/dispatch collaborators.
They never open hardware or an SDK. The highest-value test locks the
``obs_t -> control_action[0]_t`` contract: a fresh observation queried at
logical step ``t`` yields a first action encoding ``t``, published immediately
after blocking inference (not shifted by one extra control period).

The module under test imports the project's runtime stack (numpy, scipy,
pinocchio, ...). When those are unavailable in a sandbox, the whole suite is
skipped rather than reporting false passes.
"""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import dexmani_real.deployment.executor as executor_module
    from dexmani_real.deployment.executor import PolicyRunner

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    executor_module = None
    PolicyRunner = None
    _IMPORT_ERROR = exc

_STEP_DT_NS = 62_500_000  # 62.5 ms control period


class _Clock:
    """Deterministic monotonic clock used to replace ``executor.time``."""

    def __init__(self, ns: int) -> None:
        self.ns = ns

    def monotonic_ns(self) -> int:
        return self.ns

    def monotonic(self) -> float:
        return self.ns / 1e9

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)

    def advance(self, ns: int) -> None:
        self.ns += ns


class _Model:
    """Fake model whose chunk[k] encodes (logical_t + k), where logical_t is
    derived from the observation anchor time. A phase shift in the scheduler
    then surfaces as an off-by-one in the first published action value, not just
    in the recorded query anchor."""

    def __init__(self, clock: _Clock, n_action_steps: int, inference_ns: int) -> None:
        self.clock = clock
        self.n_action_steps = n_action_steps
        self.inference_ns = inference_ns

    def predict(self, observation) -> np.ndarray:
        # Blocking inference: advance the clock, then return an open-loop chunk.
        self.clock.advance(self.inference_ns)
        t = observation.anchor_monotonic_ns // _STEP_DT_NS
        return np.arange(t, t + self.n_action_steps, dtype=float).reshape(-1, 1)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"deployment runtime dependencies unavailable: {_IMPORT_ERROR}",
)
class SyncPolicyTimingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock(0)
        self.queries: list[int] = []  # anchor_ns of each _build_observation call
        self.published: list[tuple[int, tuple[float, ...]]] = []

    def _install_observation_fakes(self) -> None:
        def build(
            shared,
            policy_spec,
            *,
            observation_id,
            run_generation,
            run_started_ns,
            anchor_ns,
            step_dt_ns,
        ):
            self.queries.append(anchor_ns)
            return SimpleNamespace(
                anchor_monotonic_ns=anchor_ns, observation_id=observation_id
            )

        def to_policy(observation, policy_spec, fingertip_runtime=None):
            return SimpleNamespace(
                anchor_monotonic_ns=observation.anchor_monotonic_ns,
                observation_id=observation.observation_id,
            )

        def sources(observation):
            return {}

        def timing(observation):
            return (0.0, 0.0)

        def grid(*, run_started_ns, anchor_ns, history_len, step_dt_ns):
            return (
                np.array([anchor_ns] * history_len, dtype=np.uint64),
                anchor_ns,
            )

        patchers = [
            mock.patch.object(executor_module, "_build_observation", side_effect=build),
            mock.patch.object(executor_module, "_to_policy_observation", side_effect=to_policy),
            mock.patch.object(executor_module, "observation_sources", side_effect=sources),
            mock.patch.object(executor_module, "observation_timing_ms", side_effect=timing),
            mock.patch.object(
                executor_module, "_select_control_grid_reference_ns", side_effect=grid
            ),
            # Replace executor.time so _run_active_tick's own monotonic_ns()
            # reads (and sleeps) use the deterministic _Clock.
            mock.patch.object(executor_module, "time", self.clock),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make_runner(
        self,
        *,
        last_publication_ns=None,
        inference_ns=0,
        n_action_steps=8,
    ) -> "PolicyRunner":
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.actions = deque()
        runner.observation_id = 0
        runner.run_generation = 1
        runner.run_started_ns = 0
        runner.last_publication_ns = last_publication_ns
        runner.session_publication_count = 0
        runner.session_running_ns = 0
        runner.session_inference_ms = []
        runner.step_dt_ns = _STEP_DT_NS
        runner.chunk_sources = {}
        runner.chunk_action_index = 0
        runner.previous_arm_command_qpos = None
        runner._pending_dispatch = None
        runner._observation_waiting_since_ns = None
        runner._fifo_wait = executor_module.PublishWaitTracker("test")
        runner.stats = SimpleNamespace()
        runner.shared = None
        runner.policy_spec = SimpleNamespace(
            n_obs_steps=2,
            n_action_steps=n_action_steps,
            control_dt_s=_STEP_DT_NS / 1e9,
        )
        runner.runtime = SimpleNamespace(policy=SimpleNamespace())
        runner.model_runtime = _Model(self.clock, n_action_steps, inference_ns)
        runner.fingertip_runtime = None

        # Lifecycle / recorder stubs (always live, never expired, recorder healthy).
        runner._running_generation_is_live = lambda: True
        runner._running_time_expired = lambda now_ns: False
        runner._poll_recorder = lambda: True

        # Dispatch fake: simulate a successful publication and re-anchor cadence.
        clock = self.clock
        published = self.published

        def dispatch(action):
            published.append((clock.ns, tuple(np.asarray(action).tolist())))
            runner.last_publication_ns = clock.ns
            runner.actions.popleft()
            runner.chunk_action_index += 1

        runner._dispatch_action = dispatch
        return runner

    # --- pure helper -------------------------------------------------------

    def test_next_control_boundary_ns(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.last_publication_ns = None
        runner.step_dt_ns = _STEP_DT_NS
        self.assertIsNone(runner._next_control_boundary_ns())
        runner.last_publication_ns = 1_000_000_000
        self.assertEqual(
            runner._next_control_boundary_ns(), 1_000_000_000 + _STEP_DT_NS
        )

    # --- Policy <-> Real temporal contract (highest priority) --------------

    def test_obs_t_to_action_t_at_chunk_boundary(self):
        # Previous chunk tail published at 1000 ms. The next observation must not
        # be queried before 1062.5 ms, and its action[0] (encoding the queried
        # logical step) publishes immediately after inference — not at query+dt.
        self.clock.ns = 1_000_000_000
        self._install_observation_fakes()
        runner = self._make_runner(
            last_publication_ns=1_000_000_000, inference_ns=20_000_000
        )

        # 1062.0 ms is before the boundary: no query, no dispatch.
        self.clock.ns = 1_062_000_000
        runner._run_active_tick(self.clock.ns)
        self.assertEqual(self.queries, [])
        self.assertEqual(self.published, [])

        # At the boundary: query, 20 ms inference, immediate chunk[0] dispatch.
        self.clock.ns = 1_062_500_000
        runner._run_active_tick(self.clock.ns)

        logical_t = 1_062_500_000 // _STEP_DT_NS  # anchor-derived logical step
        self.assertEqual(self.queries, [1_062_500_000])
        # First action encodes the anchor-derived logical step and lands at
        # query+inference, not query+dt.
        self.assertEqual(self.published, [(1_082_500_000, (float(logical_t),))])

    def test_slow_inference_no_catchup(self):
        # dt=62.5 ms, inference=100 ms: new[0] lands at 1162.5 ms and new[1]
        # waits a full period after the actual publication (no catch-up burst).
        self.clock.ns = 1_000_000_000
        self._install_observation_fakes()
        runner = self._make_runner(
            last_publication_ns=1_000_000_000, inference_ns=100_000_000
        )

        self.clock.ns = 1_062_500_000
        runner._run_active_tick(self.clock.ns)
        self.assertEqual(self.published, [(1_162_500_000, (17.0,))])

        # Still before new[0]+dt: no second dispatch.
        self.clock.ns = 1_162_500_000
        runner._run_active_tick(self.clock.ns)
        self.assertEqual(len(self.published), 1)

        self.clock.ns = 1_225_000_000
        runner._run_active_tick(self.clock.ns)
        self.assertEqual(self.published[1], (1_225_000_000, (18.0,)))

    # --- unified whole-chunk invalidation ----------------------------------

    def test_invalidate_chunk_keeps_committed_reference_and_anchor(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.run_generation = 7
        runner.observation_id = 3
        runner.chunk_action_index = 2
        runner.last_publication_ns = 1_000_000_000
        runner.previous_arm_command_qpos = np.array([1.0, 2.0, 3.0])
        runner.chunk_sources = {"arm": (100, 200)}
        runner.actions = deque([np.array([1.0]), np.array([2.0])])
        runner._pending_dispatch = object()
        runner._fifo_wait = SimpleNamespace(waiting=False, note_dropped=lambda reason: None)

        runner._invalidate_chunk("test_reason")

        self.assertEqual(list(runner.actions), [])
        self.assertIsNone(runner._pending_dispatch)
        self.assertEqual(runner.chunk_sources, {})
        self.assertEqual(runner.chunk_action_index, 0)
        # A recoverable chunk drop keeps the committed continuity reference:
        # the next prediction anchors behind the last committed command, not
        # at measured qpos. Only a new epoch rebuilds the initial reference.
        self.assertTrue(
            np.array_equal(runner.previous_arm_command_qpos, np.array([1.0, 2.0, 3.0]))
        )
        # Already-occurred physical history and episode identity are preserved.
        self.assertEqual(runner.last_publication_ns, 1_000_000_000)
        self.assertEqual(runner.run_generation, 7)
        self.assertEqual(runner.observation_id, 3)

    def test_clear_execution_resets_epoch_local_state(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.run_generation = 1
        runner.last_publication_ns = 1_000_000_000
        runner.previous_arm_command_qpos = np.array([1.0])
        runner.last_recorded_action = None
        runner.actions = deque([np.array([1.0])])
        runner.chunk_sources = {"arm": (0,)}
        runner.chunk_action_index = 2
        runner.observation_id = 5
        runner._pending_dispatch = object()
        runner._observation_waiting_since_ns = 7
        # An open backpressure span must end visibly at the epoch boundary and
        # leave the tracker clean for the next trial's [WAIT].
        runner._fifo_wait = executor_module.PublishWaitTracker("test")
        runner._fifo_wait.note_full(8, 813)

        with self.assertLogs("dexmani_real.control.publication", level="WARNING") as logs:
            runner._clear_execution(None)

        self.assertEqual(runner.observation_id, 0)
        self.assertEqual(list(runner.actions), [])
        self.assertIsNone(runner._pending_dispatch)
        self.assertIsNone(runner._observation_waiting_since_ns)
        self.assertIsNone(runner.previous_arm_command_qpos)
        self.assertFalse(runner._fifo_wait.waiting)
        self.assertIn("keep_action=813", "\n".join(logs.output))
        self.assertIn("reason=epoch_boundary", "\n".join(logs.output))

    # --- lifecycle invariant -----------------------------------------------

    def test_generation_change_during_inference_discards_result(self):
        self.clock.ns = 0
        self._install_observation_fakes()
        runner = self._make_runner(last_publication_ns=None)

        state = {"live": True}
        runner._running_generation_is_live = lambda: state["live"]

        def revoke(observation):
            state["live"] = False
            return np.arange(
                observation.observation_id,
                observation.observation_id + 8,
                dtype=float,
            ).reshape(-1, 1)

        runner.model_runtime.predict = revoke

        runner._run_active_tick(0)

        self.assertEqual(self.published, [])
        self.assertEqual(list(runner.actions), [])

    # --- failure taxonomy ---------------------------------------------------

    def test_model_exception_is_session_failure_not_fault(self):
        self.clock.ns = 0
        self._install_observation_fakes()
        runner = self._make_runner(last_publication_ns=None)

        def boom(observation):
            raise RuntimeError("CUDA OOM")

        runner.model_runtime.predict = boom
        runner._invalidate_rollout = mock.Mock(return_value=True)
        runner._request_failed_session_shutdown = mock.Mock()
        runner._fault = mock.Mock()

        runner._run_active_tick(0)

        runner._invalidate_rollout.assert_called_once()
        runner._request_failed_session_shutdown.assert_called_once()
        runner._fault.assert_not_called()
        self.assertEqual(self.published, [])


if __name__ == "__main__":
    unittest.main()

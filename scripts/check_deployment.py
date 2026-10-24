"""Offline deployment regressions: synthetic clocks/models only, no device connections."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import tempfile
import threading
import time
import unittest

import numpy as np

from dexmani_real.config.execution import ExecutionConfig
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.episode import PolicyEpisode
from dexmani_real.deployment.inference import InferenceWorker, ModelResult
from dexmani_real.deployment.preflight import validate_execution_config
from dexmani_real.deployment.run import run_policy
from dexmani_real.deployment.schedule import (
    ActionPlan, PolicyQuery, control_slot, decision_expires_at_ns,
    handoff_plan, plan_rejection, prefetch_handoff, run_budget,
)


class DeploymentChecks(unittest.TestCase):
    def test_modes_rejected_before_loading(self):
        info = SimpleNamespace(control_dt_s=.0625, future_steps=15, supported_execution_modes=("sync",))
        config = ExecutionConfig(action_steps=4, execution_mode="async", prefetch_steps=1)
        with self.assertRaises(NotImplementedError):
            validate_execution_config(info, config)
        validate_execution_config(info, replace(config, execution_mode="sync"))
        info.supported_execution_modes = ("sync", "async", "rtc")
        validate_execution_config(info, config)
        with self.assertRaises(ValueError):
            validate_execution_config(info, replace(config, action_steps=15))

    def test_async_late_handoff_and_tail_expiry(self):
        query = PolicyQuery(1, 3, 4, 100, 6, np.arange(15 * 19).reshape(15, 19))
        plan, issue = handoff_plan(query, run_id=3, slot=8, mode="async", action_steps=4, now_ns=80, authorized=True)
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (4, 12))
        np.testing.assert_array_equal(plan.actions[8 - plan.anchor_slot], query.result[4])
        self.assertIsNone(plan_rejection(plan, 13, "async", 90))
        self.assertIsNotNone(plan_rejection(plan, 19, "async", 90))
        self.assertIsNotNone(plan_rejection(plan, 13, "async", 100))
        for authorized, now in ((False, 80), (True, 100)):
            accepted, issue = handoff_plan(query, run_id=3, slot=8, mode="async", action_steps=4, now_ns=now, authorized=authorized)
            self.assertIsNone(accepted)
            self.assertIsNotNone(issue)

    def test_bootstrap_sync_rtc_and_clock_boundaries(self):
        query = PolicyQuery(1, 1, 0, 100, None, np.zeros((15, 19)))
        plan, issue = handoff_plan(query, run_id=1, slot=7, mode="sync", action_steps=4, now_ns=10, authorized=True)
        self.assertIsNone(issue)
        self.assertEqual(plan.anchor_slot, 7)
        self.assertEqual(control_slot(now_ns=30, next_step_ns=10, last_slot=0, dt_ns=10, synchronous=False, resume_ready=False)[2], 2)
        query.handoff_slot = 3
        self.assertIsNotNone(handoff_plan(query, run_id=1, slot=4, mode="rtc", action_steps=4, now_ns=10, authorized=True)[1])
        self.assertEqual(decision_expires_at_ns([90], 100, 10), 101)
        self.assertIsNone(decision_expires_at_ns([90], 101, 10))
        self.assertIsNone(decision_expires_at_ns([102], 101, 10))
        self.assertEqual(run_budget(run_id=1, started_ns=0, max_running_s=1, wait_started_ns=0, max_wait_s=1), (10**9, "wait"))
        self.assertEqual(prefetch_handoff(plan, slot=10, mode="async", prefetch_steps=1,
                                        query_pending=True, model_pending=True, history_ready=True), (None, None))

    def test_retired_query_is_not_applied_to_next_run(self):
        episode = PolicyEpisode.__new__(PolicyEpisode)
        episode.run_id = 2
        episode.query = PolicyQuery(7, 1, 0, 100, None)
        episode.results = Mock()
        episode.termination = object()
        episode.has_motion_authority = lambda: True
        episode.model = SimpleNamespace(poll=lambda: ("predict", ModelResult(np.ones((4, 19)), None, 1, 2, dict(run_id=1, query_id=7))))
        episode.poll_model()
        self.assertIsNone(episode.query)
        episode.results.record_inference.assert_not_called()

    def test_worker_single_inflight_and_serial_close(self):
        entered, release = threading.Event(), threading.Event()
        operations = []
        class Model:
            def warmup(self, **kwargs):
                operations.append("warmup")
            def predict(self, observation):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("synthetic worker not released")
                operations.append("predict")
                return np.zeros((4, 19))
            def close(self):
                operations.append("close")
        worker = InferenceWorker(Model, build_observation=lambda rows: rows)
        self.addCleanup(release.set)
        self.addCleanup(worker.close)
        worker.submit("load")
        worker.future.result(timeout=2)
        self.assertIsNone(worker.poll()[1].error)
        worker.submit("predict", ())
        self.assertTrue(entered.wait(1))
        with self.assertRaises(RuntimeError):
            worker.submit("reset_episode")
        worker.close()
        self.assertEqual(worker.close_result()["status"], "pending")
        release.set()
        result = worker.close_result(timeout_s=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(operations, ["warmup", "predict", "close"])

    def test_startup_failure_publishes_required_result_without_devices(self):
        from dexmani_real.deployment.session import PolicySession
        from dexmani_real.runtime.observation import ObservationNeeds

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = SimpleNamespace(checkpoint_path=Path("model.pt"), weights="raw",
                                   inference_steps=10, control_dt_s=.0625)
            session = PolicySession(
                ExperimentConfig(), info, control_needs=ObservationNeeds(),
                record_needs=ObservationNeeds(), results_dir=root, run_config={},
            )
            with patch.object(session, "prepare_resources", side_effect=RuntimeError("model failed")), \
                 patch.object(session, "start_devices") as devices:
                self.assertEqual(session.run(), 1)
            devices.assert_not_called()
            result = json.loads((root / "session_result.json").read_text())
            self.assertEqual(result["outcome"], "fault")
            self.assertEqual(result["reason"], "model failed")
            self.assertEqual(result["episodes"], [])
            self.assertTrue(result["shutdown_clean"])

    def test_result_required_raw_optional_and_output_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = SimpleNamespace(
                n_action_steps=4, observation_fields=("joint_state",), policy_name="dp3",
                task_name="synthetic", experiment_dir=root / "exp", future_steps=15,
                control_dt_s=.0625, to_dict=lambda: {"synthetic": True},
            )
            cfg = {"eval": {"use_ema": True, "inference_steps": 10, "action_steps": 1}}
            with patch("dexmani_policy.deployment.metadata.resolve_experiment", return_value=root), \
                 patch("dexmani_policy.common.config.load_experiment_config", return_value=cfg), \
                 patch("dexmani_policy.common.experiments.resolve_checkpoint_path", return_value=root / "checkpoint.pt"), \
                 patch("dexmani_policy.deployment.inspect_policy", return_value=info), \
                 patch("dexmani_real.deployment.run.validate_deployment_config", return_value=(None, None)), \
                 patch("dexmani_real.deployment.run.resolve_table_plane", return_value=None), \
                 patch("dexmani_real.deployment.run.PolicySession") as session:
                session.return_value.run.return_value = 0
                for raw in (True, False):
                    self.assertEqual(run_policy("synthetic", ExperimentConfig(), record_raw=raw, output=root), 0)
                    kwargs = session.call_args.kwargs
                    self.assertEqual(session.call_args.args[0].execution.action_steps, info.n_action_steps)
                    self.assertIsNotNone(kwargs["results_dir"])
                    self.assertTrue((kwargs["results_dir"] / "run_config.yaml").is_file())
                    self.assertEqual(kwargs["recording_dir"] is not None, raw)
                session.reset_mock()
                with patch("dexmani_real.deployment.run.atomic_yaml_dump", side_effect=OSError("disk full")), self.assertRaises(OSError):
                    run_policy("synthetic", ExperimentConfig(), output=root)
                session.assert_not_called()


if __name__ == "__main__":
    unittest.main()

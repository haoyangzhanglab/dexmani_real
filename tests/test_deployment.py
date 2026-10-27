import json
import threading
import unittest
from dataclasses import asdict, replace
from itertools import count, product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml

from dexmani_real.config import load_experiment_config
from dexmani_real.control.actions import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.deployment import rollout, run, session
from dexmani_real.deployment.config import ExecutionConfig, resolve_deployment_config
from dexmani_real.deployment.inference import InferenceWorker, ModelResult
from dexmani_real.deployment.observation import (
    build_policy_observation,
    policy_dependencies,
    policy_needs,
    project_policy_row,
)
from dexmani_real.deployment.results import SessionResults
from dexmani_real.deployment.schedule import PolicyQuery, control_slot, handoff_plan
from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_observation,
)
from dexmani_real.runtime.state import RunEndReason, SessionState


def policy_info(**changes):
    return SimpleNamespace(**{
        "n_obs_steps": 2, "n_action_steps": 2, "horizon": 6, "future_steps": 5,
        "hand_history_steps": None, "action_mode": "joint", "control_dt_s": 0.05,
        "observation_fields": ("joint_state",), "supported_execution_modes": ("sync", "async", "rtc"),
        "checkpoint_path": Path("checkpoint.pt"), "weights": "ema", "inference_steps": 10,
        "task_name": "test", **changes,
    })


class ConfigTests(unittest.TestCase):
    def test_saved_defaults_and_explicit_execution_override(self):
        runtime = load_experiment_config()
        info = policy_info()
        resolved = resolve_deployment_config(info, runtime, ObservationNeeds())
        self.assertIsNone(runtime.execution.action_steps)
        self.assertEqual(resolved.execution.action_steps, 2)
        configured = replace(runtime, execution=ExecutionConfig(action_steps=1))
        self.assertEqual(resolve_deployment_config(info, configured, ObservationNeeds()), configured)

    def test_cloud_recipe_is_restored_without_changing_declared_config(self):
        runtime = load_experiment_config()
        recipe = {**asdict(runtime.pointcloud), "num_points": 16}
        info = policy_info(observation_fields=("point_cloud",), pointcloud_config=recipe)
        resolved = resolve_deployment_config(info, runtime, ObservationNeeds(point_cloud=True))
        self.assertEqual(resolved.pointcloud.num_points, 16)
        self.assertEqual(runtime.pointcloud.num_points, 1024)

    def test_invalid_windows_and_lamp_mode_fail_before_startup(self):
        runtime = load_experiment_config()
        for config in (
            ExecutionConfig(action_steps=0),
            ExecutionConfig(action_steps=True),
            ExecutionConfig(execution_mode="async"),
            ExecutionConfig(action_steps=5, execution_mode="rtc", prefetch_steps=1, rtc_guidance_cap=0),
            ExecutionConfig(execution_mode="rtc", prefetch_steps=1, rtc_guidance_cap=float("nan")),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_deployment_config(policy_info(), replace(runtime, execution=config), ObservationNeeds())
        with self.assertRaisesRegex(ValueError, "LAMP"):
            resolve_deployment_config(policy_info(hand_history_steps=3), runtime, ObservationNeeds())


class RunTests(unittest.TestCase):
    def test_entry_delegates_policy_resolution_and_preserves_run_config(self):
        runtime = load_experiment_config()
        runtime = replace(runtime, environment=replace(
            runtime.environment, table=replace(runtime.environment.table, enabled=False),
        ))
        recipe = {
            **asdict(runtime.pointcloud), "num_points": 16, "remove_table": False,
            "workspace": list(runtime.pointcloud.workspace),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            checkpoint = experiment / "checkpoints" / "epoch=4-step=80-milestone=80pct.pt"
            info = policy_info(
                experiment_dir=experiment, checkpoint_path=checkpoint, policy_name="test_policy",
                observation_fields=("point_cloud",), pointcloud_config=recipe,
                to_dict=lambda: {"policy_name": "test_policy", "observation_fields": ["point_cloud"]},
            )
            selectors = ("latest", "best", "80pct", checkpoint.name, str(checkpoint))
            for record_raw, selector in product((False, True), selectors):
                with (
                    self.subTest(record_raw=record_raw, selector=selector), patch("builtins.print"),
                    patch("dexmani_policy.deployment.inspect_policy", return_value=info) as inspect,
                    patch.object(run, "git_head", return_value=None),
                    patch.object(run, "resolve_table_plane", return_value=None) as table,
                    patch.object(run, "PolicySession") as owner,
                ):
                    owner.return_value.run.return_value = 0
                    self.assertEqual(run.run_policy(
                        str(experiment), runtime, checkpoint=selector, output=root, record_raw=record_raw,
                    ), 0)
                    inspect.assert_called_once_with(
                        str(experiment), checkpoint=selector, weights=None, inference_steps=None,
                    )
                    resolved = owner.call_args.args[0]
                    self.assertEqual(resolved.execution.action_steps, info.n_action_steps)
                    self.assertEqual(resolved.pointcloud.num_points, 16)
                    self.assertFalse(table.call_args.kwargs["required"])
                    kwargs = owner.call_args.kwargs
                    self.assertEqual(kwargs["record_raw"], record_raw)
                    self.assertEqual(kwargs["record_needs"].camera, "rgbd" if record_raw else "none")
                    config = yaml.safe_load((kwargs["output_dir"] / "run_config.yaml").read_text())
                    self.assertEqual(set(config), {
                        "experiment", "checkpoint", "policy", "future_steps", "runtime",
                        "camera_calibration_path", "effective_camera", "table_plane_abcd",
                        "table_plane_path", "seed", "device", "num_episodes", "max_duration_s",
                        "output", "record_raw", "dexmani_real_head", "dexmani_policy_head",
                    })
                    self.assertEqual(config["runtime"]["pointcloud"], recipe)
                    self.assertEqual(config["record_raw"], record_raw)
                    self.assertEqual(config["checkpoint"], selector)
                    self.assertEqual(config["output"], str(kwargs["output_dir"]))


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.now = 10_000_000_000
        clock = patch.object(rollout.time, "monotonic_ns", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.runtime = load_experiment_config()
        self.state = SessionState()
        self.state.arm()
        self.state.stop_signal.clear()
        self.robot = Mock(stop_required=False)
        self.model = Mock(has_pending_task=False)
        self.completions = []
        self.model.submit.side_effect = lambda *args, **kwargs: setattr(self.model, "has_pending_task", True)

        def poll():
            if not self.completions:
                return None
            self.model.has_pending_task = False
            return self.completions.pop(0)

        self.model.poll.side_effect = poll
        self.mapper = Mock()
        self.mapper.map_joint.side_effect = lambda arm, hand, *args: ActionResult(arm + 0.01, hand)
        self.mapper.control_from_command.side_effect = lambda cmd, mode: np.r_[cmd.arm_qpos, cmd.hand_qpos]
        self.mapper.frozen_rejection.return_value = None
        self.results = Mock()
        reader = patch.object(rollout, "read_observation", side_effect=lambda *a, **kw: self.row())
        reader.start()
        self.addCleanup(reader.stop)

    def row(self):
        arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
        arm["qpos"] = self.runtime.arm.home_qpos
        hand["qpos"] = np.deg2rad(self.runtime.hand.home_qpos_deg)
        arm["timestamp_ns"] = hand["timestamp_ns"] = self.now
        hand["tactile_aggregate_timestamp_ns"] = self.now - 2
        hand["tactile_dense_timestamp_ns"] = self.now - 3
        return Observation(
            arm, hand, {"timestamp_ns": self.now - 4, "rgb": np.zeros((2, 2, 3), np.uint8)},
            None, np.zeros((16, 6), np.float32), self.now, self.now - 5,
        )

    def make_rollout(self, mode="sync", **info_changes):
        info = policy_info(**info_changes)
        runtime = replace(self.runtime, execution=ExecutionConfig(
            action_steps=info.n_action_steps, execution_mode=mode, prefetch_steps=1, rtc_guidance_cap=0,
        ))
        dependencies = policy_dependencies(info.observation_fields)
        return rollout.PolicyRollout(
            self.state, runtime, info, preparing_epoch=self.state.run_id,
            robot=self.robot, sensors=Mock(channels=None), action_mapper=self.mapper,
            model=self.model, results=self.results, control_needs=policy_needs(dependencies),
            record_needs=ObservationNeeds(), dependencies=dependencies, max_running_s=60,
            query_ids=count(1), capture_ids=count(1),
        )

    def test_rejected_joint_candidate_does_not_become_a_command(self):
        owner = self.make_rollout()
        self.mapper.map_joint.side_effect = None
        self.mapper.map_joint.return_value = ActionResult(
            np.zeros(7), np.zeros(12), rejection_reason="collision",
        )
        command, rejection = owner.map_command(np.zeros(19), self.row(), None)
        self.assertIsNone(command)
        self.assertIn("collision", rejection.detail)
        self.robot.send_action.assert_not_called()

    def complete(self, value=None, error=None):
        call = self.model.submit.call_args
        self.completions.append((call.args[0], ModelResult(
            value, error, self.now - 2, self.now,
            query_context=call.kwargs["query_context"], model_started_ns=self.now - 1,
        )))

    def start(self, mode="sync", **changes):
        item = self.make_rollout(mode, **changes)
        item.prepare()
        self.complete()
        item.poll_model()
        self.assertIsNotNone(item.run_id)
        self.model.submit.reset_mock()
        return item

    def tick(self, item):
        self.now = item.next_step_ns
        item.step()

    def bootstrap(self, mode, **changes):
        item = self.start(mode, **changes)
        self.tick(item)
        self.model.submit.assert_not_called()
        self.tick(item)
        self.complete(np.zeros((5, 19)))
        self.tick(item)
        self.assertEqual(self.robot.send_action.call_count, 1)
        return item

    def test_first_prediction_waits_for_history_and_owns_snapshot(self):
        item = self.start()
        self.tick(item)
        self.model.submit.assert_not_called()
        self.tick(item)
        rows = self.model.submit.call_args.args[1]
        observation = build_policy_observation(rows, ("joint_state",), item.dependencies)
        self.assertEqual(observation["joint_state"].shape, (2, 19))
        self.assertEqual(observation["joint_state"].dtype, np.float32)
        self.tick(item)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[-1].observation_timestamp_ns, item.history[-1].observation_timestamp_ns)
        self.assertFalse(rows[-1].values["hand_qpos"].flags.writeable)

    def test_async_keeps_old_tail_until_late_prediction_takes_over(self):
        item = self.bootstrap("async")
        self.tick(item)
        self.assertEqual(self.model.submit.call_count, 2)
        self.tick(item)
        self.assertEqual(self.robot.send_action.call_count, 3)
        self.complete(np.full((5, 19), 0.2))
        self.tick(item)
        self.assertIsNone(item.termination)
        np.testing.assert_allclose(self.robot.send_action.call_args.args[0].hand_qpos, 0.2)

    def test_rtc_freezes_mapped_commands_and_stops_on_missing_handoff(self):
        item = self.bootstrap("rtc")
        self.tick(item)
        query = self.model.submit.call_args
        prefix = query.kwargs["rtc_prefix"]
        np.testing.assert_allclose(prefix[0, :7], self.robot.send_action.call_args.args[0].arm_qpos)
        self.assertEqual(query.kwargs["delay_steps"], 1)
        self.assertEqual(self.mapper.map_joint.call_count, 2)
        self.tick(item)
        self.assertEqual(item.termination.detail, "handoff_result_not_ready")
        self.assertEqual(self.robot.send_action.call_count, 2)
        self.robot.stop.assert_called_once()

    def test_completed_prefetch_remains_owned_until_handoff(self):
        item = self.bootstrap("async", n_action_steps=3)
        item.execution = replace(item.execution, prefetch_steps=2)
        self.tick(item)
        self.complete(np.full((5, 19), 0.2))
        self.tick(item)
        self.assertFalse(self.model.has_pending_task)
        self.assertIsNotNone(item.query.result)
        self.assertEqual(self.model.submit.call_count, 2)
        np.testing.assert_allclose(self.robot.send_action.call_args.args[0].hand_qpos, 0.0)
        self.tick(item)
        np.testing.assert_allclose(self.robot.send_action.call_args.args[0].hand_qpos, 0.2)

    def test_sync_waits_between_chunks_and_can_resume_before_polling_deadline(self):
        item = self.bootstrap("sync")
        self.tick(item)
        self.assertIsNone(item.plan)
        self.tick(item)
        self.assertEqual(self.robot.send_action.call_count, 2)
        self.complete(np.zeros((5, 19)))
        self.now += 1
        item.step()
        self.assertEqual(self.robot.send_action.call_count, 3)

    def test_stop_discards_late_prediction_and_keeps_first_termination(self):
        item = self.start(n_obs_steps=1)
        self.tick(item)
        self.state.request_stop()
        item.step()
        end = item.termination
        self.complete(np.zeros((5, 19)))
        item.step()
        self.assertIs(item.termination, end)
        self.assertEqual(end.reason, "operator")
        self.robot.send_action.assert_not_called()
        self.results.finish_episode.assert_called_once()

    def test_late_model_failure_faults_session_without_rewriting_episode(self):
        item = self.start(n_obs_steps=1)
        self.tick(item)
        self.state.request_stop()
        item.step()
        self.complete(error=ValueError("model failed"))
        with self.assertLogs(rollout.logger, level="ERROR"), self.assertRaisesRegex(RuntimeError, "retired"):
            item.step()
        self.assertTrue(self.state.error_state)
        self.results.finish_episode.assert_called_once()
        self.results.record_errors.assert_called_once()

    def test_lamp_context_advances_only_after_successful_submission(self):
        item = self.start(n_obs_steps=1, n_action_steps=1, hand_history_steps=3)
        self.tick(item)
        snapshot = self.model.submit.call_args.kwargs["action_context"]["hand_history"]
        initial = snapshot.copy()
        self.complete(np.full((1, 19), 0.2))
        self.tick(item)
        np.testing.assert_allclose(item.hand_history[-1], 0.2)
        np.testing.assert_array_equal(snapshot, initial)
        self.tick(item)
        self.complete(np.full((1, 19), 0.4))
        self.robot.send_action.side_effect = TimeoutError("send expired")
        self.tick(item)
        self.assertEqual(item.termination.detail, "send expired")
        np.testing.assert_array_equal(snapshot, initial)

    def test_source_timestamp_uses_requested_modalities_and_validates_upstream(self):
        for fields, offset in (
            (("joint_state",), 0), (("rgb",), 4), (("point_cloud",), 5),
            (("finger_tactile_force",), 2), (("taxel_tactile_force",), 3),
            (("camera_intrinsic",), 0), (("rgb", "point_cloud", "joint_state"), 5),
        ):
            with self.subTest(fields=fields):
                dependencies = policy_dependencies(fields)
                projected = project_policy_row(self.row(), fields, dependencies)
                self.assertEqual(projected.source_timestamp_ns, self.now - offset)
        for stamp in (0, self.now + 1):
            row = self.row()
            row.arm["timestamp_ns"] = stamp
            issues, _ = check_observation(row, self.runtime, ObservationNeeds(), now_ns=self.now)
            self.assertIn("arm_feedback_stale", issues)

    def test_policy_input_boundary_still_rejects_nonfinite_observations(self):
        from dexmani_policy.deployment.runtime import PolicyRuntime

        info = policy_info(n_obs_steps=1, rgb_preprocessing={})
        policy = PolicyRuntime(
            SimpleNamespace(consumed_observation_fields=info.observation_fields),
            info, device="cpu", seed=0,
        )
        dependencies = policy_dependencies(info.observation_fields)
        row = self.row()
        row.hand["qpos"][0, 0] = np.nan
        observation = build_policy_observation(
            (project_policy_row(row, info.observation_fields, dependencies),),
            info.observation_fields, dependencies,
        )
        with self.assertRaisesRegex(ValueError, "Nonfinite policy observation"):
            policy.prepare_observations(observation)

    def test_decision_age_includes_the_exact_budget_boundary(self):
        item = self.start(n_obs_steps=1)
        item.history.append(project_policy_row(self.row(), ("joint_state",), item.dependencies))
        self.now += item.max_decision_age_ns
        self.assertIsNone(item.submit(0))
        self.assertEqual(item.query.expires_at_ns, self.now + 1)
        self.results.record_duration.assert_called_with("source_age", item.max_decision_age_ns)
        self.now += 1
        self.assertEqual(item.submit(0).detail, "query_sources_invalid_or_expired")

    def test_stop_and_recording_failures_still_finalize_once(self):
        item = self.start(n_obs_steps=1)
        item.recorder = Mock(start_attempted=True)
        item.recorder.append.side_effect = RecordingError("append failed")
        item.recorder.finish.side_effect = RecordingError("finish failed")
        item.pending_frame = (self.row(), None)
        error = RobotError("stop failed")
        self.robot.stop.side_effect = error
        with self.assertRaises(RobotError) as raised:
            item.finish("test", run_end_reason=RunEndReason.OPERATOR)
        self.assertIs(raised.exception, error)
        item.finish("again")
        item.recorder.finish.assert_called_once()
        self.results.finish_episode.assert_called_once()
        errors = self.results.finish_episode.call_args.kwargs["errors"]
        self.assertEqual([e["stage"] for e in errors], ["stop", "recording", "recorder_finalize"])


class ScheduleTests(unittest.TestCase):
    def test_skipped_slots_never_generate_catchup_actions(self):
        self.assertEqual(control_slot(
            now_ns=130, next_step_ns=100, last_slot=1, dt_ns=10,
            synchronous=False, resume_ready=False,
        ), (2, 100, 3))

    def test_handoff_preserves_anchor_and_rejects_revocation_and_expiry(self):
        query = PolicyQuery(1, 1, 2, 100, 4, np.zeros((5, 19)))
        kwargs = dict(run_id=1, slot=5, mode="async", action_steps=2, now_ns=99, authorized=True)
        plan, issue = handoff_plan(query, **kwargs)
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (2, 7))
        self.assertEqual(handoff_plan(query, **{**kwargs, "now_ns": 100})[1].detail,
                         "handoff_decision_expired")
        self.assertEqual(handoff_plan(query, **{**kwargs, "authorized": False})[1].detail,
                         "handoff_authority_revoked")


class InferenceTests(unittest.TestCase):
    def test_warmup_and_close_failures_are_both_reported(self):
        model = Mock()
        model.warmup.side_effect = ValueError("warmup failed")
        model.close.side_effect = RuntimeError("close failed")
        worker = InferenceWorker(lambda: model, build_observation=lambda rows: rows)
        self.addCleanup(lambda: worker.close().result(timeout=2))
        worker.submit("load")
        with self.assertLogs("dexmani_real.deployment.inference", level="ERROR"):
            worker.close()
            result = worker.close_result(timeout_s=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retired_task"]["operation"], "load")
        self.assertEqual(result["retired_task"]["message"], "warmup failed")
        self.assertEqual(result["close_error"], {
            "exception_type": "RuntimeError", "message": "close failed",
        })
        model.close.assert_called_once()

    def test_operations_are_serial_and_retired_failure_does_not_skip_close(self):
        started, release = threading.Event(), threading.Event()
        threads, operations = [], []

        def operation(name):
            threads.append(threading.get_ident())
            operations.append(name)

        def predict(observation):
            operation("predict")
            started.set()
            if not release.wait(2):
                raise TimeoutError("test prediction was not released")
            raise ValueError("late failure")

        model = Mock()
        model.warmup.side_effect = lambda: operation("warmup")
        model.reset_episode.side_effect = lambda: operation("reset")
        model.predict.side_effect = predict
        model.close.side_effect = lambda: operation("close")
        worker = InferenceWorker(lambda: model, build_observation=lambda rows: rows)
        self.addCleanup(lambda: worker.close().result(timeout=2))
        self.addCleanup(release.set)
        for name in ("load", "reset_episode"):
            worker.submit(name)
            worker.future.result(timeout=1)
            self.assertIsNone(worker.poll()[1].error)
        context = {"run_id": 1, "query_id": 2}
        worker.submit("predict", (), query_context=context)
        self.assertTrue(started.wait(1))
        with self.assertRaises(RuntimeError):
            worker.submit("reset_episode")
        worker.close()
        self.assertEqual(worker.close_result()["status"], "pending")
        with self.assertLogs("dexmani_real.deployment.inference", level="ERROR"):
            release.set()
            result = worker.close_result(timeout_s=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retired_task"]["query_context"], context)
        self.assertEqual(result["retired_task"]["message"], "late failure")
        self.assertEqual(operations, ["warmup", "reset", "predict", "close"])
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())


class SessionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.session = session.PolicySession(
            load_experiment_config(), policy_info(), control_needs=ObservationNeeds(),
            record_needs=ObservationNeeds(), dependencies=frozenset(), output_dir=self.output,
            run_config={}, device="cpu", seed=0, max_running_s=60, num_episodes=20,
            record_raw=False, camera_calibration_path=None, table_plane_abcd=None,
        )

    def test_warmup_provides_home_hand_history_only_when_policy_requires_it(self):
        for history_steps in (None, 3):
            with self.subTest(history_steps=history_steps):
                owner = self.session
                owner.info = policy_info(hand_history_steps=history_steps)
                owner.model = Mock()
                owner.model.poll.return_value = ("load", ModelResult([], None, 0, 1))
                owner.warmup_policy()
                context = owner.model.submit.call_args.kwargs["action_context"]
                if history_steps is None:
                    self.assertIsNone(context)
                else:
                    history = context["hand_history"]
                    self.assertEqual(history.shape, (history_steps, 12))
                    self.assertEqual(history.dtype, np.float32)
                    expected = np.deg2rad(owner.runtime.hand.home_qpos_deg).astype(np.float32)
                    np.testing.assert_array_equal(history, np.broadcast_to(expected, history.shape))

    def test_startup_failure_publishes_fault_without_creating_devices(self):
        owner = self.session
        owner.prepare_resources = Mock(side_effect=ValueError("invalid calibration"))
        owner.start_devices = Mock()
        with (patch.object(session, "configure_logging"), patch("builtins.print"),
              self.assertLogs(session.logger, level="ERROR")):
            self.assertEqual(owner.run(), 1)
        owner.start_devices.assert_not_called()
        result = json.loads((self.output / "session_result.json").read_text())
        self.assertEqual(result["outcome"], "fault")
        self.assertEqual(result["reason"], "invalid calibration")
        self.assertEqual(result["episodes"], [])
        self.assertEqual(result["model_close"]["reason"], "model_not_created")

    def test_session_keeps_primary_failure_when_episode_cleanup_also_fails(self):
        owner = self.session
        state = SessionState()
        primary = RobotError("hardware failed")
        owner.prepare_resources = lambda: setattr(owner, "runtime_state", state)
        owner.start_devices = Mock()
        owner.run_control_loop = Mock(side_effect=primary)
        owner.advance_episode = Mock(side_effect=RecordingError("writer failed"))
        owner.close = Mock(return_value=1)
        with (patch.object(session, "configure_logging"), patch("builtins.print"),
              self.assertLogs(session.logger, level="ERROR")):
            self.assertEqual(owner.run(), 1)
        self.assertIs(owner.close.call_args.kwargs["failure"], primary)
        self.assertEqual(owner.advance_episode.call_args.kwargs["reason"], RunEndReason.HARDWARE_FAULT)
        self.assertTrue(state.error_state)

    def test_close_updates_raw_and_preserves_session_result_fields(self):
        owner = self.session
        owner.results = SessionResults(self.output)
        owner.results.begin_episode()
        owner.results.finish_episode("operator", raw={"status": "pending"})
        owner.runtime_state = SessionState()
        owner.robot = Mock(stop_required=False)
        owner.sensors = Mock()
        owner.model = Mock()
        owner.model.close_result.return_value = {"status": "completed", "reason": None}
        owner.recorder = Mock()
        owner.recorder.close.side_effect = RecordingError("writer failed")
        owner.recorder.result.return_value = {"status": "published"}
        owner.recorder_result_index = 1
        with patch.object(session, "close_runtime", return_value=True) as close:
            self.assertEqual(owner.close(clean=True, failure=None), 1)
        close.assert_called_once()
        result = json.loads((self.output / "session_result.json").read_text())
        self.assertEqual(set(result), {
            "kind", "started_utc", "ended_utc", "outcome", "reason", "shutdown_clean",
            "episodes", "errors", "model_close", "home_results",
        })
        self.assertEqual(result["reason"], "shutdown_failed")
        self.assertEqual(result["errors"][0]["stage"], "recorder_close")
        self.assertEqual(result["episodes"][0]["termination_reason"], "operator")
        self.assertEqual(result["episodes"][0]["raw"], {"status": "published"})


if __name__ == "__main__":
    unittest.main()

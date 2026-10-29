"""真实公开接口、启动观测和部署时钟的无设备回归。"""

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from dexmani_policy.deployment import PolicyInfo, inspect_policy, load_policy
from dexmani_real.config.deployment import ACTION_TIMEOUT_NS
from dexmani_real.config.real import load_config
from dexmani_real.deployment.inference import PolicyWorker, PredictionResult
from dexmani_real.deployment.observation import ObservationBuilder, policy_dependencies, policy_needs
from dexmani_real.deployment.preflight import resolve_deployment_config
from dexmani_real.deployment.rollout import EpisodeResult, Rollout
from dexmani_real.deployment.run import session_directory
from dexmani_real.deployment.schedule import Prediction
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import STATUS_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation, ObservationNeeds
from dexmani_real.runtime.state import SafetyState, SessionState


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def advance(self, ns):
        self.now += ns


def observation(now):
    arm = np.zeros(1, dtype=STATUS_DTYPE)["frame"]
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm["timestamp_ns"] = now
    for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
        hand[name] = now
    hand["qpos"] = .2
    return Observation(arm, hand, None, None, None, now)


class DeploymentTests(unittest.TestCase):
    def info(self, directory, *, mode="joint", history=None, n_obs=1):
        path = Path(directory)
        (path / "checkpoints").mkdir(exist_ok=True)
        (path / "checkpoints/latest.pt").touch()
        cfg = dict(
            policy_name="test", task_name="test", action_key="action" if mode == "joint" else "action_ee",
            eval=dict(use_ema=False, inference_steps=1),
            agent=dict(n_obs_steps=n_obs, n_action_steps=1, horizon=n_obs, action_dim=19 if mode == "joint" else 21),
            dataset=dict(sensor_modalities=["joint_state", "finger_tactile_force"] if history else ["joint_state"]),
            tactile_source="real" if history else None,
            policy_metadata=dict(action_start_index=n_obs - 1, control_action_dim=19 if mode == "joint" else 21,
                                 supported_execution_modes=["sync"],
                                 action_context_shapes={} if history is None else {"hand_history": [history, 12]}),
            deployment_metadata=dict(control_dt_s=.05),
        )
        (path / "config.yaml").write_text(yaml.safe_dump(cfg))
        return inspect_policy(path, checkpoint="latest")

    def test_real_public_info_to_warmup_and_predict(self):
        for mode, history in (("joint", 3), ("eef", None)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                info = self.info(directory, mode=mode, history=history)
                self.assertIsInstance(info, PolicyInfo)
                dependencies = policy_dependencies(info.observation_fields)
                needs = policy_needs(dependencies)
                runtime = resolve_deployment_config(info, load_config(), needs)
                builder = ObservationBuilder(info, runtime, dependencies=dependencies, needs=needs)
                calls = []

                class Agent:
                    control_action_dim = info.control_action_dim

                    def predict_action(self, tensors, **kwargs):
                        calls.append(threading.get_ident())
                        if history is not None:
                            torch.testing.assert_close(kwargs["action_context"]["hand_history"],
                                                       torch.full((1, history, 12), .2))
                        return {"future_actions": torch.zeros((1, 1, info.control_action_dim))}

                with patch("dexmani_policy.deployment.runtime.restore_policy", return_value=Agent()):
                    worker = PolicyWorker(lambda: load_policy(info, device="cpu"), build_observation=builder.build)
                    session = PolicySession.__new__(PolicySession)
                    session.info, session.runtime = info, runtime
                    session.runtime_state = SessionState()
                    session.keyboard = SimpleNamespace(estop_latched=False, check_health=lambda: True)
                    ready_calls = []
                    session.robot = SimpleNamespace(service_feedback=lambda: None,
                                                    reset_hand_feedback=lambda: ready_calls.append("reset"),
                                                    read_initial_state=lambda **kwargs: ready_calls.append(kwargs),
                                                    read_state=lambda: observation(time.monotonic_ns()))
                    session.sensors = SimpleNamespace(channels=None)
                    session.control_needs, session.observations, session.model = needs, builder, worker
                    try:
                        worker.load()
                        session.warmup_policy()
                        self.assertEqual(ready_calls[0], "reset")
                        self.assertEqual(ready_calls[1]["required_tactile"], ("aggregate",) if history else ())
                        self.assertEqual(session.runtime_state.safety_state, SafetyState.DISARMED)
                        worker.reset()
                        session.wait_model(time.monotonic() + 2.)
                        rows = (builder.project(observation(time.monotonic_ns())),)
                        context = None if history is None else {"hand_history": np.full((history, 12), .2)}
                        worker.predict(rows, action_context=context)
                        operation, result = session.wait_model(time.monotonic() + 2.)
                        self.assertEqual(operation, "predict")
                        self.assertEqual(result.actions.shape, (1, info.control_action_dim))
                        self.assertEqual(len(set(calls)), 1)
                        self.assertNotEqual(calls[0], threading.get_ident())
                    finally:
                        worker.close()
                        self.assertEqual(worker.wait_closed(2.), ("completed", []))

    def test_recipe_validation_uses_required_numbers_only(self):
        with tempfile.TemporaryDirectory() as directory:
            info = self.info(directory)
            runtime = load_config()
            cloud_info = replace(info, observation_fields=("joint_state", "point_cloud", "hand_pointcloud"),
                                 pointcloud_config=asdict(runtime.pointcloud) | {"note": "provenance"},
                                 hand_pointcloud_config=asdict(runtime.hand_pointcloud) | {"note": "provenance"})
            resolved = resolve_deployment_config(cloud_info, runtime, ObservationNeeds())
            self.assertEqual(resolved.pointcloud, runtime.pointcloud)
            missing = dict(cloud_info.pointcloud_config)
            del missing["workspace"]
            with self.assertRaisesRegex(ValueError, "saved numerical"):
                resolve_deployment_config(replace(cloud_info, pointcloud_config=missing), runtime, ObservationNeeds())
            with self.assertRaisesRegex(ValueError, "control_dt_s"):
                resolve_deployment_config(replace(info, control_dt_s=None), runtime, ObservationNeeds())

    def make_rollout(self, directory, clock, *, n_obs=1):
        info = self.info(directory, n_obs=n_obs)
        runtime = resolve_deployment_config(info, load_config(), ObservationNeeds())
        state = SessionState()
        state.arm()
        builder = ObservationBuilder(info, runtime, dependencies=policy_dependencies(info.observation_fields),
                                     needs=ObservationNeeds())
        model = SimpleNamespace(has_pending_task=False, predict=lambda *args, **kwargs: None,
                                poll=lambda: None)
        robot = SimpleNamespace(stop_required=False, prepare_streaming=lambda epoch: state.stop_signal.clear(),
                                stop=lambda: None, read_state=lambda: observation(clock.now), send_action=lambda *a, **k: None)
        rollout = Rollout(state, runtime, info, preparing_epoch=state.run_id, robot=robot,
                          observations=builder, record_needs=ObservationNeeds(), needs=ObservationNeeds(),
                          sensors=SimpleNamespace(channels=None), action_mapper=None, model=model,
                          max_running_s=60., capture_ids=iter(range(1, 100)), index=1,
                          output_dir=Path(directory), record_raw=False)
        rollout.start_observation = lambda: observation(clock.now)
        rollout.admit()
        rollout.read_observation = lambda: (observation(clock.now), None, clock.now + 100_000_000)
        return rollout

    def test_sync_results_receive_a_full_period(self):
        for phase in (10_000_000, 5_000_000):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                clock = Clock()
                with patch("time.monotonic_ns", side_effect=lambda: clock.now):
                    rollout = self.make_rollout(directory, clock)
                    rollout.step()  # 首个观测提交预测，保留 sync_row。
                    old_tick = rollout.next_step_ns
                    clock.now = old_tick - phase
                    rollout.query.result = np.zeros((1, 19))
                    sent = []

                    def map_command(row, slot):
                        clock.advance(6_000_000)
                        return RobotCommand(rollout.run_id, np.zeros(7), np.zeros(12)), None

                    rollout.command_for_slot = map_command
                    rollout.robot.send_action = lambda command, **kwargs: sent.append(kwargs)
                    rollout.step()
                    self.assertFalse(rollout.finished)
                    self.assertEqual(len(sent), 1)
                    self.assertGreater(sent[0]["valid_until_ns"], clock.now)
                    self.assertEqual(rollout.result.metrics["observation_to_submit_count"], 1)
                    self.assertAlmostEqual(rollout.result.metrics["observation_to_submit_mean_s"],
                                           (clock.now - rollout.started_ns) / 1e9)

    def test_long_history_has_separate_prediction_budget_and_total_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            with patch("time.monotonic_ns", side_effect=lambda: clock.now):
                rollout = self.make_rollout(directory, clock, n_obs=42)
                original_deadline = rollout.run_deadline_ns
                self.assertEqual(rollout.wait_deadline_ns, clock.now + 41 * rollout.dt_ns + ACTION_TIMEOUT_NS)
                calls = []
                rollout.model.predict = lambda rows, **kwargs: calls.append(rows)
                for index in range(42):
                    clock.now = rollout.started_ns + index * rollout.dt_ns
                    rollout.step()
                self.assertEqual(len(calls), 1)
                self.assertEqual(len({row.observation_timestamp_ns for row in calls[0]}), 42)
                self.assertEqual(rollout.wait_deadline_ns, clock.now + ACTION_TIMEOUT_NS)
                self.assertEqual(rollout.run_deadline_ns, original_deadline)
                # 实验总截止期仍先于预测等待截止时立即终止。
                rollout.run_deadline_ns = clock.now
                rollout.step()
                self.assertEqual(rollout.result.termination_details, "duration_timeout")

    def test_late_prediction_is_retired_and_directory_creation_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            with patch("time.monotonic_ns", side_effect=lambda: clock.now):
                rollout = self.make_rollout(directory, clock)
                rollout.model.has_pending_task = True
                rollout.query = Prediction(rollout.run_id, 0, None, clock.now, clock.now)
                rollout.stop("cancel")
                rollout.model.poll = lambda: ("predict", PredictionResult(np.zeros((1, 19)), 1))
                rollout.poll_model()
                self.assertIsNone(rollout.query)
                self.assertIsNone(rollout.plan)
            with patch("dexmani_real.deployment.run.time.strftime", return_value="same"):
                with ThreadPoolExecutor(2) as executor:
                    paths = list(executor.map(lambda _: session_directory(Path(directory), rollout.policy_info), range(2)))
            self.assertNotEqual(*paths)
            session = PolicySession.__new__(PolicySession)
            session.output_dir = Path(directory)
            session.summary, session.errors = {"outcome": "finished"}, []
            session.episodes = [EpisodeResult(1, started_utc="now", termination_reason="timeout",
                                             termination_details="duration_timeout"),
                                EpisodeResult(2, started_utc="now", termination_reason="executor_boundary"),
                                EpisodeResult(3, termination_reason="start_cancelled", errors=[{"message": "late error"}])]
            session.save_results()
            result = json.loads((Path(directory) / "session_result.json").read_text())
            self.assertEqual(result["outcome"], "finished")
            self.assertEqual(result["started_episode_count"], 2)
            self.assertEqual(result["technical_termination_count"], 2)


if __name__ == "__main__":
    unittest.main()

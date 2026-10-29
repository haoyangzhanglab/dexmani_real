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
from unittest.mock import Mock, patch

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

    def camera_session(self, directory, fields, *, record_raw, calibration_path):
        session = PolicySession.__new__(PolicySession)
        session.runtime = load_config()
        session.info = SimpleNamespace(observation_fields=fields, depth_config={"depth_scale_m": .001})
        dependencies = policy_dependencies(fields)
        control_needs = policy_needs(dependencies)
        session.needs = control_needs.union(
            ObservationNeeds(camera_fields=("rgb", "depth")) if record_raw else ObservationNeeds()
        )
        camera = session.runtime.camera
        channels = SimpleNamespace(
            camera_ready=SimpleNamespace(is_set=lambda: True),
            pointcloud_ready=None,
            camera_serial=SimpleNamespace(value=b"selected"),
            camera_depth_scale=SimpleNamespace(value=.001),
            camera_intrinsics=SimpleNamespace(value=json.dumps(dict(
                width=camera.width, height=camera.height, fx=1, fy=1, ppx=0, ppy=0,
                distortion_model="none", distortion_coeffs=[0] * 5,
            )).encode()),
        )
        session.record_raw, session.camera_calibration_path = record_raw, calibration_path
        session.runtime_state = SessionState()
        session.robot, session.keyboard = Mock(), Mock()
        session.observations = ObservationBuilder(
            session.info, session.runtime, dependencies=dependencies, needs=control_needs,
        )
        session.sensors = SimpleNamespace(channels=channels, start=Mock(), shutdown=Mock(return_value=True))
        session.process_context = SimpleNamespace(Process=Mock())
        session.output_dir, session.run_config = Path(directory), {}
        session.table_plane_abcd = None
        session.warmup_policy = Mock()
        return session

    def test_deployment_reads_extrinsics_only_for_consumers_or_recording(self):
        from dexmani_real.utils.camera_calibrate import extrinsics

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            fields = ("rgb", "depth", "camera_intrinsic")
            cases = (({}, KeyError), ({"selected": {"T_base_from_camera": "broken"}}, ValueError))
            for contents, error in cases:
                path.write_text(json.dumps(contents))
                with self.subTest(contents=contents, consumer="rgb/depth/intrinsic"):
                    session = self.camera_session(directory, fields, record_raw=False, calibration_path=path)
                    session.start_devices()
                    self.assertIsNone(session.camera_snapshot.T_xarm_base_from_color)
                    self.assertEqual(session.runtime_state.safety_state, SafetyState.ARMED)
                    self.assertEqual(set(session.observations.camera_parameters), {"camera_intrinsic"})
                for required in (("point_cloud",), ("camera_extrinsic",)):
                    with self.subTest(contents=contents, consumer=required):
                        session = self.camera_session(directory, required, record_raw=False, calibration_path=path)
                        with self.assertRaises(error):
                            session.start_devices()
                        session.warmup_policy.assert_not_called()
                with self.subTest(contents=contents, recording=True):
                    session = self.camera_session(directory, fields, record_raw=True, calibration_path=path)
                    with self.assertRaises(error):
                        session.start_devices()

            transform = np.eye(4)
            transform[0, 3] = .4
            path.write_text(json.dumps({"selected": {"T_base_from_camera": transform.tolist()}}))
            for requested, record_raw in ((("point_cloud",), False), (("camera_extrinsic",), False), (fields, True)):
                with self.subTest(consumer=requested, recording=record_raw):
                    session = self.camera_session(directory, requested, record_raw=record_raw, calibration_path=path)
                    session.start_devices()
                    np.testing.assert_array_equal(session.camera_snapshot.T_xarm_base_from_color, transform)
                    self.assertEqual(session.run_config["effective_camera"]["camera_T_xarm_base_from_color"],
                                     transform.ravel().tolist())

            missing = Path(directory) / "missing.json"
            for record_raw in (False, True):
                with self.subTest(default_calibration_missing=True, recording=record_raw):
                    session = self.camera_session(directory, fields, record_raw=record_raw, calibration_path=None)
                    with patch.object(extrinsics, "CAMERAS_PATH", missing):
                        session.start_devices()
                    self.assertIsNone(session.camera_snapshot.T_xarm_base_from_color)
            for required in (("point_cloud",), ("camera_extrinsic",)):
                with self.subTest(default_calibration_missing=True, consumer=required):
                    session = self.camera_session(directory, required, record_raw=False, calibration_path=None)
                    with patch.object(extrinsics, "CAMERAS_PATH", missing), self.assertRaises(FileNotFoundError):
                        session.start_devices()

    def test_dataset_metadata_to_saved_config_and_public_inspect(self):
        import zarr
        from omegaconf import OmegaConf

        from dexmani_policy.training.common.config import assemble_effective_config
        from dexmani_policy.training.common.data import build_dataset
        from dexmani_policy.training.common.logging import ExperimentLogger
        from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.zarr"
            runtime = load_config(overrides={
                "pointcloud": {"num_points": 8, "remove_table": False},
                "hand_pointcloud": {"num_points": 16},
            })
            attrs = dict(
                format="dexmani.real.canonical", source_id="integration-source", dt=.05,
                depth_scale_m_per_unit=.001,
                pointcloud_config=asdict(runtime.pointcloud),
                hand_pointcloud_config=asdict(runtime.hand_pointcloud),
                fingertip_link_names=list(XHAND_FINGERTIP_LINK_NAMES),
                processing_source={"repository": "legacy-constant"},
            )
            store = zarr.open_group(str(data_path), mode="w")
            store.attrs.update(attrs)
            attrs = dict(zarr.open_group(str(data_path), mode="r").attrs)
            arrays = store.create_group("data")
            for name, shape in dict(
                joint_state=(2, 19), action=(2, 19), depth=(2, *runtime.camera.depth_shape),
                point_cloud=(2, runtime.pointcloud.num_points, 6),
                hand_pointcloud=(2, runtime.hand_pointcloud.num_points, 6),
                fingertip_xyz=(2, 5, 3), finger_tactile_force=(2, 5, 3),
            ).items():
                arrays.create_dataset(name, data=np.zeros(shape, dtype=np.uint16 if name == "depth" else np.float32))
            store.create_group("meta").create_dataset("episode_ends", data=np.array([2], dtype=np.int64))
            modalities = [name for name in arrays if name != "action"]
            cfg = OmegaConf.create(dict(
                policy_name="test", task_name="test", action_key="action", tactile_source="real",
                eval=dict(use_ema=False, inference_steps=1),
                agent=dict(n_obs_steps=1, n_action_steps=1, horizon=1, action_dim=19),
                normalization=(
                    {name: "identity" for name in modalities} | {"action": "limits", "joint_state": "limits"}
                ),
                training=dict(lr_scheduler="cosine"),
                dataset=dict(
                    _target_="dexmani_policy.datasets.common.robot_dataset.RobotDataset",
                    zarr_path=str(data_path), sensor_modalities=modalities,
                    horizon=1, obs_horizon=1, val_ratio=0, tactile_source="real",
                ),
            ))
            dataset, description = build_dataset(cfg, verbose=False)
            model = SimpleNamespace(
                inference_metadata=dict(
                    action_start_index=dataset.action_start_index, control_action_dim=19,
                    supported_execution_modes=["sync"], action_context_shapes={"hand_history": [3, 12]},
                ),
                effective_config={},
            )
            effective = assemble_effective_config(cfg, description, model)
            experiment = root / "experiment"
            logger = ExperimentLogger(experiment)
            try:
                logger.save_config(effective)
            finally:
                logger.close()
            (experiment / "checkpoints").mkdir()
            (experiment / "checkpoints/latest.pt").touch()  # inspect 只选择路径，不读取权重。
            info = inspect_policy(experiment, checkpoint="latest")
            self.assertEqual(info.control_dt_s, attrs["dt"])
            self.assertEqual(info.depth_config, dict(shape=list(runtime.camera.depth_shape),
                                                    depth_scale_m=attrs["depth_scale_m_per_unit"]))
            self.assertEqual(info.pointcloud_config, attrs["pointcloud_config"])
            self.assertEqual(info.hand_pointcloud_config, attrs["hand_pointcloud_config"])
            self.assertEqual(info.fingertip_link_names, tuple(attrs["fingertip_link_names"]))
            self.assertEqual(info.tactile_source, "real")
            self.assertEqual(info.hand_history_steps, 3)
            saved = yaml.safe_load((experiment / "config.yaml").read_text())
            self.assertEqual(saved["dataset"]["split_manifest"]["source_id"], attrs["source_id"])
            self.assertNotIn("processing_source", saved["deployment_metadata"])
            needs = policy_needs(policy_dependencies(info.observation_fields))
            resolved = resolve_deployment_config(info, load_config(), needs)
            self.assertEqual(resolved.pointcloud, runtime.pointcloud)
            self.assertEqual(resolved.hand_pointcloud, runtime.hand_pointcloud)
            self.assertEqual(resolved.camera.depth_shape, runtime.camera.depth_shape)
            self.assertEqual(resolved.execution.action_steps, info.n_action_steps)

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
            for mode, dimension in (("joint", 18), ("eef", 19)):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "control_action_dim"):
                    resolve_deployment_config(
                        replace(info, action_mode=mode, control_action_dim=dimension), runtime, ObservationNeeds(),
                    )

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

    def test_async_and_rtc_keep_prediction_time_anchors_across_handoffs(self):
        for mode in ("async", "rtc"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                clock = Clock()
                with patch("time.monotonic_ns", side_effect=lambda: clock.now):
                    rollout = self.make_rollout(directory, clock)
                    rollout.execution = replace(rollout.execution, mode=mode, action_steps=2,
                                                prefetch_steps=1, rtc_guidance_cap=1.0)
                    rollout.policy_info = replace(rollout.policy_info, horizon=6, future_steps=6,
                                                  n_action_steps=2, supported_execution_modes=(mode,))
                    pending, calls, submitted = [], [], []

                    def predict(rows, **kwargs):
                        self.assertFalse(pending)
                        calls.append((rows[-1].observation_timestamp_ns, kwargs))
                        pending.append((clock.now + rollout.dt_ns // 2, np.zeros((6, 19))))
                        rollout.model.has_pending_task = True

                    def poll():
                        if pending and clock.now >= pending[0][0]:
                            _, actions = pending.pop()
                            rollout.model.has_pending_task = False
                            return "predict", PredictionResult(actions, rollout.dt_ns // 2)

                    rollout.model.predict, rollout.model.poll = predict, poll
                    rollout.robot.check = lambda: None
                    rollout.map_command = lambda *args, **kwargs: (
                        RobotCommand(rollout.run_id, np.zeros(7), np.zeros(12)), None)
                    rollout.action_mapper = SimpleNamespace(
                        frozen_rejection=lambda *args: None,
                        control_from_command=lambda command, mode: np.r_[command.arm_qpos, command.hand_qpos],
                    )
                    rollout.robot.send_action = lambda command, **kwargs: submitted.append(
                        (clock.now, rollout.plan.observation_ns))
                    for tick in range(7):
                        clock.now = rollout.started_ns + tick * rollout.dt_ns
                        rollout.step()
                        self.assertFalse(rollout.finished)
                    self.assertEqual(len(submitted), 6)
                    expected_sources = [calls[index // 2][0] for index in range(6)]
                    self.assertEqual([source for _, source in submitted], expected_sources)
                    expected_latency = np.mean([
                        (now - expected_sources[index]) / 1e9 for index, (now, _) in enumerate(submitted)
                    ])
                    metrics = rollout.result.metrics
                    self.assertEqual(metrics["inference_count"], 3)
                    self.assertAlmostEqual(metrics["inference_mean_s"], .025)
                    self.assertAlmostEqual(metrics["prediction_wait_mean_s"], .05)
                    self.assertAlmostEqual(metrics["observation_to_submit_mean_s"], expected_latency)
                    for _, kwargs in calls[1:]:
                        self.assertEqual("rtc_prefix" in kwargs, mode == "rtc")
                        if mode == "rtc":
                            self.assertEqual(kwargs["delay_steps"], 1)

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

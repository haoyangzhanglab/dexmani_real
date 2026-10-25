"""使用合成输入、假设备和真实 FK 验证部署边界，不连接设备。"""

import io
import json
import multiprocessing as mp
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.execution import ExecutionConfig
from dexmani_real.config.experiment import ExperimentConfig, load_experiment_config
from dexmani_real.data.recorder import RecordingError
from dexmani_real.data.raw.schema import encode_raw_row
from dexmani_real.deployment.episode import PolicyEpisode
from dexmani_real.deployment.inference import InferenceWorker, ModelResult
from dexmani_real.deployment.observation import (
    build_observation_camera,
    build_observation_kinematics,
    build_policy_observation,
    policy_dependencies,
    project_policy_row,
)
from dexmani_real.deployment.preflight import validate_execution_config
from dexmani_real.deployment.run import run_policy
from dexmani_real.deployment.schedule import (
    ActionPlan, PolicyQuery, control_slot, decision_expires_at_ns,
    handoff_plan, plan_rejection, prefetch_handoff, run_budget,
)
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import make_camera_frame_dtype, make_pointcloud_frame_dtype
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.robot.commands import RobotCommand, RobotError, OperationCancelled
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import (
    ObservationNeeds, ObservationRow, check_observation, control_deadline_ns,
    read_observation, snapshot_camera_parameters,
)
from dexmani_real.runtime.state import RuntimeState, RunEndReason
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, CameraSnapshot


def observation_row(stamp=1_000_000_000):
    runtime = ExperimentConfig()
    arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
    arm["qpos"], hand["qpos"] = runtime.arm.home_qpos, np.deg2rad(runtime.hand.home_qpos_deg)
    arm["timestamp_ns"] = stamp
    for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
        hand[name] = stamp
    camera = dict(timestamp_ns=stamp, ring_sequence=1,
                  rgb=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.uint16))
    return ObservationRow(
        arm, hand, camera, dict(recv_ts_ns=stamp), np.ones((2, 6), np.float32), stamp,
        pointcloud_timestamp_ns=stamp, control_timestamp_ns=stamp,
    )


def running_episode(*, recorder=None):
    state = RuntimeState()
    state.arm()
    state.stop_signal.clear()
    run_id, _ = state.begin_motion()
    runtime = ExperimentConfig(execution=ExecutionConfig(action_steps=4))
    info = SimpleNamespace(n_obs_steps=2, hand_history_steps=None, control_dt_s=.0625,
                           observation_fields=("joint_state",), action_mode="joint")
    episode = PolicyEpisode(
        state, runtime, info, preparing_epoch=run_id, robot=Mock(stop_required=False),
        control_needs=ObservationNeeds(), record_needs=ObservationNeeds(camera="rgbd"),
        dependencies=policy_dependencies(info.observation_fields),
        sensors=SimpleNamespace(channels=None), action_mapper=Mock(),
        model_runtime=SimpleNamespace(has_pending_task=False, poll=lambda: None, submit=Mock()),
        max_running_s=60, query_ids=count(1), capture_ids=count(1),
        recorder=recorder, results=Mock(),
    )
    episode.run_id = run_id
    episode.started_ns = episode.next_step_ns = 1_000_000_000
    return episode


class DeploymentChecks(unittest.TestCase):
    def test_camera_snapshot_admission_and_float32_conversion(self):
        color = CameraIntrinsics(16, 16, 100., 100., 7.5, 7.5, "none", (0.,) * 5)
        transform = np.array([
            [0., -1., 0., 1.], [1., 0., 0., 2.], [0., 0., 1., 3.], [0., 0., 0., 1.],
        ])
        snapshot = CameraSnapshot("synthetic", color, .001, transform)
        transform[:] = np.nan
        with patch("dexmani_real.sensor.camera.geometry.validate_rigid_transform",
                   side_effect=AssertionError("snapshot geometry revalidated")):
            parameters = build_observation_camera(snapshot, ("camera_intrinsic", "camera_extrinsic"))
        np.testing.assert_array_equal(parameters["camera_extrinsic"], np.array([
            0., 1., 0., -2., -1., 0., 0., 1., 0., 0., 1., -3.,
        ], dtype=np.float32))
        self.assertFalse(snapshot.T_xarm_base_from_color.flags.writeable)
        for invalid in (np.eye(3), np.full((4, 4), np.nan), np.diag([2., 1., 1., 1.])):
            with self.assertRaises(ValueError):
                CameraSnapshot("synthetic", color, .001, invalid)
        overflow = np.eye(4)
        overflow[0, 3] = 1e40
        snapshot = CameraSnapshot("synthetic", color, .001, overflow)
        with self.assertRaisesRegex(ValueError, "float32"):
            build_observation_camera(snapshot, ("camera_extrinsic",))
        tiny_focal = replace(color, fx=1e-50)
        snapshot = CameraSnapshot("synthetic", tiny_focal, .001, None)
        with self.assertRaisesRegex(ValueError, "positive in float32"):
            build_observation_camera(snapshot, ("camera_intrinsic",))
        channels = SimpleNamespace(
            camera_ready=SimpleNamespace(is_set=lambda: True),
            camera_serial=SimpleNamespace(value=b"synthetic"),
            camera_intrinsics=SimpleNamespace(value=json.dumps(color.to_dict()).encode()),
            camera_depth_scale=SimpleNamespace(value=.001),
        )
        config = replace(ExperimentConfig().camera, width=16, height=16)
        with self.assertRaisesRegex(ValueError, "extrinsics"):
            snapshot_camera_parameters(channels, config, None, require_extrinsics=True)
        snapshot = snapshot_camera_parameters(channels, config, None)
        self.assertEqual(build_observation_camera(
            snapshot, ("depth",), depth_config={"depth_scale_m": .001},
        ), {})
        with self.assertRaisesRegex(ValueError, "depth scale"):
            build_observation_camera(snapshot, ("depth",), depth_config={"depth_scale_m": .002})
        self.assertEqual(build_observation_camera(None, ("joint_state",)), {})

    def test_projection_and_window_cache_with_real_fk(self):
        runtime = ExperimentConfig()
        fields = ("joint_state", "eef_pose", "fingertip_xyz", "finger_tactile_force")
        dependencies = policy_dependencies(fields)
        kinematics = build_observation_kinematics(
            dependencies, fingertip_link_names=runtime.hand.fingertip_link_names,
            hand=runtime.hand, hand_pointcloud=runtime.hand_pointcloud,
        )
        physical = [observation_row(1_000_000_000 + i * 62_500_000) for i in range(5)]
        rows = [project_policy_row(row, fields, dependencies) for row in physical]
        self.assertEqual(set(rows[0].values), {"arm_qpos", "hand_qpos", "finger_tactile_force"})
        for key in rows[0].values:
            self.assertIsNone(rows[0].values[key].base)
            self.assertFalse(rows[0].values[key].flags.writeable)
        expected = [build_policy_observation((row,), fields, dependencies, kinematics=kinematics)
                    for row in rows]
        kinematics.row_features.clear()
        with patch.object(kinematics.arm_fk, "compute", wraps=kinematics.arm_fk.compute) as fk, \
             patch("dexmani_real.deployment.observation.policy_dependencies", side_effect=AssertionError("hot-path dependency resolution")):
            for start in range(3):
                result = build_policy_observation(rows[start:start + 3], fields, dependencies, kinematics=kinematics)
                self.assertEqual(len(kinematics.row_features), 3)
                for field in fields:
                    np.testing.assert_array_equal(result[field], np.concatenate([
                        value[field] for value in expected[start:start + 3]
                    ]))
            self.assertEqual(fk.call_count, 5)
            kinematics.row_features.clear()
            build_policy_observation(rows[:3], fields, dependencies, kinematics=kinematics)
            self.assertEqual(fk.call_count, 8)
        with patch.object(kinematics.hand_fk.pin, "forwardKinematics",
                          side_effect=AssertionError("invalid joints reached native FK")):
            for invalid in (np.zeros(11), np.full(12, np.nan)):
                with self.assertRaises(ValueError):
                    kinematics.hand_fk.compute_frame_placements(
                        invalid, kinematics.hand_fk.fingertip_frame_ids,
                    )
            for invalid in (np.zeros(6), np.full(7, np.nan)):
                with self.assertRaises(ValueError):
                    kinematics.arm_fk.compute(invalid)
        overflow = replace(kinematics, handbase_position_eef_m=(1e40, 0., 0.), row_features={})
        with np.errstate(over="ignore"), self.assertRaisesRegex(RuntimeError, "finite float32"):
            build_policy_observation(rows[:1], fields, dependencies, kinematics=overflow)
        # The decision deadline belongs to the latest query row, not the oldest history row.
        episode = running_episode()
        latest = replace(rows[-1], source_times=(rows[0].source_times[0] + 2_000_000_000,),
                         observation_timestamp_ns=rows[0].observation_timestamp_ns + 2_000_000_000)
        self.assertIsNone(decision_expires_at_ns(rows[0].source_times,
                                               latest.observation_timestamp_ns, episode.max_decision_age_ns))
        with patch("time.monotonic_ns", return_value=latest.observation_timestamp_ns):
            self.assertIsNone(episode.submit((rows[0], latest), 0))
        self.assertEqual(episode.query.expires_at_ns,
                         latest.source_times[0] + episode.max_decision_age_ns + 1)

    def test_core_imports_keep_backends_lazy_and_adapter_import_does_not_connect(self):
        probe = '''
import importlib
import importlib.abc
import sys
from types import ModuleType
class RejectBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xarm', 'xhand_controller', 'pyrealsense2', 'serial', 'pynput', 'torch'}:
            raise AssertionError('unexpected backend import: ' + fullname)
sys.meta_path.insert(0, RejectBackend())
for name in ('deployment.run', 'teleop.session', 'replay.session', 'robot.xhand.driver',
             'robot.xarm7.worker', 'sensor.camera.worker', 'sensor.vr.worker', 'data.canonical.export'):
    importlib.import_module('dexmani_real.' + name)
def reject_device(*args, **kwargs):
    raise AssertionError('unexpected device access')
sdk = ModuleType('pyrealsense2')
sdk.context = sdk.pipeline = sdk.config = sdk.frame_queue = sdk.align = reject_device
sys.modules['pyrealsense2'] = sdk
importlib.import_module('dexmani_real.sensor.camera.realsense')
'''
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_declared_config_and_print_without_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("execution:\n  action_steps: 3\n  execution_mode: async\n")
            runtime = load_experiment_config(yaml_path=config, cli_overrides={"execution.action_steps": 4})
            self.assertEqual((runtime.execution.action_steps, runtime.execution.execution_mode), (4, "async"))
            for contents in (
                'hand:\n  enabled: "false"\n',
                "execution:\n  unknown: 1\n",
                "environment:\n  table:\n    plane_abcd: [0, 0, 1, 0]\n",
            ):
                config.write_text(contents)
                with self.assertRaises(TypeError):
                    load_experiment_config(yaml_path=config)
        with patch.object(DexManiRobot, "connect", side_effect=AssertionError("device connection")), \
             patch.dict("sys.modules", {name: None for name in ("xarm", "xhand_controller", "pyrealsense2", "torch", "dexmani_policy")}):
            for script in ("collect_teleop.py", "run_policy.py"):
                entry = runpy.run_path(str(Path(__file__).with_name(script)))
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(entry["main"](["--print-config"]), 0)
                self.assertIn("arm:", output.getvalue())
                self.assertNotIn("no_results", output.getvalue())

    def test_direct_hand_driver_rejects_invalid_targets_before_sdk(self):
        from dexmani_real.robot.xhand.driver import XHand

        hand = XHand(ExperimentConfig().hand)
        hand.control = Mock()
        for target in (np.zeros(11), np.full(12, np.nan), np.full(12, 100.0)):
            with self.assertRaises(ValueError):
                hand.send_position(target)
        hand.control.send_command.assert_not_called()
        episode = running_episode()
        state, runtime = episode.runtime_state, episode.runtime
        robot = DexManiRobot(runtime.arm, runtime.hand, run_signal=state.run_signal,
                             stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
                             signal_lock=state.motion_lock)
        robot.owner, robot.connected = threading.get_ident(), True
        robot.arm = Mock(is_connected=True)
        for target in (np.zeros(6), np.full(7, np.nan), np.full(7, 100.)):
            with self.assertRaises(ValueError):
                robot.send_action(RobotCommand(state.run_id, target), valid_until_ns=10, execute_until_ns=20)
        robot.arm.submit.assert_not_called()

    def test_source_boundaries_and_independent_recording_needs(self):
        runtime, row = ExperimentConfig(), observation_row()
        for source, needs, budget in (
            ("arm", ObservationNeeds(), runtime.arm.feedback_max_age_s),
            ("camera", ObservationNeeds(camera="rgbd"), runtime.camera.max_frame_age_s),
            ("pointcloud", ObservationNeeds(point_cloud=True), runtime.camera.max_frame_age_s),
            ("vr", ObservationNeeds(vr=True), runtime.teleop.vr_mapping.stale_threshold_s),
        ):
            for delta, stale in ((-1, False), (0, False), (1, True)):
                now = row.observation_timestamp_ns + int(budget * 1e9) + delta
                issues, _ = check_observation(row, runtime, needs, require_hand=False, now_ns=now)
                self.assertEqual(any(issue.startswith(source) for issue in issues), stale)
        now = row.observation_timestamp_ns + int(runtime.hand.feedback_max_age_s * 1e9)
        self.assertIn("hand_timestamp_ns_unavailable_or_stale",
                      check_observation(row, runtime, ObservationNeeds(), now_ns=now)[0])
        self.assertTrue(check_observation(row, runtime, ObservationNeeds(), now_ns=1)[0])
        self.assertEqual(check_observation(replace(row, hand=None), runtime, ObservationNeeds(),
                                          require_hand=False, now_ns=row.observation_timestamp_ns)[1],
                         row.observation_timestamp_ns + int(runtime.arm.feedback_max_age_s * 1e9) + 1)
        # 接收相机年龄上界仍通过，但此时提交期限已到；不能统一这两个比较符。
        camera_stamp = row.observation_timestamp_ns - int(runtime.camera.max_frame_age_s * 1e9)
        row.camera["timestamp_ns"] = camera_stamp
        issues, deadline = check_observation(row, runtime, ObservationNeeds(camera="rgbd"),
                                            now_ns=row.observation_timestamp_ns)
        self.assertEqual(issues, [])
        self.assertEqual(deadline, row.observation_timestamp_ns)
        self.assertEqual(control_deadline_ns(deadline, runtime.arm, now_ns=deadline + 1), deadline)
        for missing in ("camera", "aggregate", "dense"):
            episode = running_episode(recorder=Mock())
            episode.record_needs = ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
            row = observation_row()
            if missing == "camera":
                row = replace(row, camera=None)
            else:
                row.hand[f"tactile_{missing}_timestamp_ns"] = 0
            with patch("dexmani_real.deployment.episode.read_observation", return_value=row), \
                 patch("time.monotonic_ns", return_value=row.observation_timestamp_ns):
                with self.assertRaisesRegex(RecordingError, "recording sources"):
                    episode.read_observation()
            self.assertIsNone(episode.pending_frame)
            episode.robot.arm.update_health.assert_not_called()

    def test_pointcloud_pairing_overwrite_and_owned_snapshot(self):
        ctx = mp.get_context("spawn")
        camera = SharedMemoryRingBuffer(ctx, make_camera_frame_dtype((4, 4, 3), (4, 4)), 2)
        cloud = SharedMemoryRingBuffer(ctx, make_pointcloud_frame_dtype(2), 2)
        source = observation_row()
        header = np.zeros((), dtype=camera.dtype["header"])
        header["timestamp_ns"] = source.observation_timestamp_ns
        camera.write(header=header, rgb=source.camera["rgb"], depth=source.camera["depth"])
        payload = np.zeros((), dtype=cloud.dtype)
        payload["source_camera_sequence"] = 1
        payload["timestamp_ns"] = source.observation_timestamp_ns
        payload["point_cloud"] = 2
        cloud.write(**{name: payload[name] for name in cloud.dtype.names})
        robot = Mock(read_state=Mock(return_value=SimpleNamespace(arm=source.arm, hand=source.hand)))
        channels = SimpleNamespace(camera_ring=camera, pointcloud_ring=cloud)
        needs = ObservationNeeds(camera="rgbd", point_cloud=True)
        row = read_observation(channels, robot, needs=needs)
        self.assertEqual(row.camera["ring_sequence"], int(payload["source_camera_sequence"]))
        for value in (3, 4):
            camera.write(header=header, rgb=np.full((4, 4, 3), value, np.uint8), depth=source.camera["depth"])
        missing = read_observation(channels, robot, needs=needs)
        self.assertIsNone(missing.camera)
        self.assertEqual(missing.pointcloud_timestamp_ns, source.observation_timestamp_ns)
        np.testing.assert_array_equal(row.camera["rgb"], np.zeros((4, 4, 3), np.uint8))
        self.assertFalse(row.camera["rgb"].flags.writeable)
        cached = cloud.read_latest(use_last_good=True)
        cloud.slots[int(cloud.header[0])]["sequence"] -= 1
        self.assertIs(cloud.read_latest(use_last_good=True), cached)
        self.assertEqual(int(cached[0]["timestamp_ns"][0]), source.observation_timestamp_ns)

    def test_finish_first_cause_pending_once_and_stop_failure_is_cleanup(self):
        for stop_fails, enqueue_fails in ((False, False), (True, False), (False, True)):
            events = []
            recorder = Mock(start_attempted=True, staging_path=None, written_frames=1)
            episode = running_episode(recorder=recorder)
            row = observation_row()
            command = RobotCommand(episode.run_id, row.arm["qpos"][0], row.hand["qpos"][0])
            episode.pending_frame = (row, command)
            def stop():
                self.assertFalse(episode.has_motion_authority())
                events.append("stop")
                if stop_fails:
                    raise RobotError("synthetic stop failure")
            def enqueue(*args):
                self.assertIsNone(episode.pending_frame)
                self.assertIs(args[1], command)
                events.append("enqueue")
                if enqueue_fails:
                    raise RecordingError("synthetic queue failure")
            episode.robot.stop.side_effect = stop
            recorder.add_frame.side_effect = enqueue
            recorder.save_episode.side_effect = lambda **kwargs: events.append("save")
            episode.results.finish_episode.side_effect = lambda *args, **kwargs: events.append("result")
            with patch("time.monotonic_ns", return_value=1_010_000_000):
                if stop_fails or enqueue_fails:
                    with self.assertRaises((RobotError, RecordingError)):
                        episode.finish("operator", run_end_reason=RunEndReason.OPERATOR)
                else:
                    episode.finish("operator", run_end_reason=RunEndReason.OPERATOR)
                first = episode.termination
                episode.finish("later", run_end_reason=RunEndReason.HARDWARE_FAULT)
            self.assertIs(episode.termination, first)
            self.assertEqual(events, ["stop", "enqueue", "save", "result"])
            self.assertEqual(recorder.save_episode.call_args.kwargs["technical_status"],
                             "failed" if enqueue_fails else "normal")
            self.assertEqual(first.reason, "operator")

    def test_nonatomic_submission_checks_between_devices(self):
        for failure in ("cancel", "deadline", "hand", "after_hand"):
            state = RuntimeState()
            runtime = ExperimentConfig()
            robot = DexManiRobot(runtime.arm, runtime.hand, run_signal=state.run_signal,
                                 stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
                                 signal_lock=state.motion_lock)
            robot.owner, robot.connected = threading.get_ident(), True
            robot.arm, robot.hand = Mock(is_connected=True), Mock(is_connected=True)
            row, events, now = observation_row(), [], [100]
            command = RobotCommand(state.run_id, row.arm["qpos"][0], row.hand["qpos"][0])
            def arm_submit(*args, **kwargs):
                events.append("arm")
                if failure == "cancel":
                    state.run_signal.value += 1
                if failure == "deadline":
                    now[0] = 200
            def hand_send(*args):
                events.append("hand")
                if failure == "hand":
                    raise OSError("synthetic SDK failure")
                state.stop_signal.set()
            robot.arm.submit.side_effect = arm_submit
            robot.hand.send_action.side_effect = hand_send
            with patch("time.monotonic_ns", side_effect=lambda: now[0]), \
                 self.assertRaises((OperationCancelled, TimeoutError, RobotError)):
                robot.send_action(command, valid_until_ns=200, execute_until_ns=300)
            self.assertEqual(events, ["arm"] if failure in {"cancel", "deadline"} else ["arm", "hand"])
            self.assertTrue(state.stop_signal.is_set())
            encoded = encode_raw_row(row, command, 0.0)
            np.testing.assert_array_equal(encoded["action_arm_joint_target"], command.arm_qpos)
            self.assertTrue(np.isnan(encode_raw_row(row, None, 0.0)["action_arm_joint_target"]).all())

    def test_arm_publication_preserves_run_input_sequence_and_operation_identity(self):
        from dexmani_real.robot.xarm7.executor import ArmExecutor
        from dexmani_real.robot.xarm7.protocol import STATUS_DTYPE, LifecycleReply, OperationStatus

        state, config = RuntimeState(), ExperimentConfig().arm
        executor = ArmExecutor(config, run_signal=state.run_signal, stop_signal=state.stop_signal,
                               emergency_signal=state.emergency_signal, signal_lock=state.motion_lock)
        self.addCleanup(executor.pipe.close)
        self.addCleanup(executor.child_pipe.close)
        executor.process = Mock(is_alive=lambda: True)
        executor.io.connected.set()
        executor.io.input_generation.value = 5
        status = np.zeros((), STATUS_DTYPE)
        status["prepared_run_id"] = state.run_id
        executor.io.publish(status)
        command = RobotCommand(state.run_id, config.home_qpos, input_generation=5)
        with patch("time.monotonic_ns", return_value=100):
            executor.submit(command, submit_until_ns=200, execute_until_ns=300)
            with self.assertRaisesRegex(RuntimeError, "slot is full"):
                executor.submit(command, submit_until_ns=200, execute_until_ns=300)
            goal = executor.io.take_goal()
            self.assertEqual((goal["sequence"], goal["run_id"], goal["input_generation"]), (1, state.run_id, 5))
            for invalid in (replace(command, input_generation=4), replace(command, run_id=state.run_id + 1)):
                with self.assertRaises(OperationCancelled):
                    executor.submit(invalid, submit_until_ns=200, execute_until_ns=300)
            with self.assertRaises(TimeoutError):
                executor.submit(command, submit_until_ns=100, execute_until_ns=300)
            self.assertEqual(executor.sequence, 1)
            self.assertIsNone(executor.io.take_goal())
            executor.submit(command, submit_until_ns=200, execute_until_ns=300)
            self.assertEqual(executor.io.take_goal()["sequence"], 2)
        executor.child_pipe.send(LifecycleReply(3, OperationStatus.COMPLETED))
        with self.assertRaisesRegex(RuntimeError, "reply mismatch"):
            executor.receive_reply(4)
        executor.child_pipe.send(LifecycleReply(4, OperationStatus.COMPLETED))
        self.assertEqual(executor.receive_reply(4).operation_id, 4)

    def test_arm_worker_rechecks_authority_and_deadline_after_computation(self):
        from dexmani_real.robot.xarm7.executor import ArmChannels
        from dexmani_real.robot.xarm7.protocol import GOAL_DTYPE
        from dexmani_real.robot.xarm7.worker import ArmWorker

        for change in ("none", "revoke", "expire"):
            state, config, now = RuntimeState(), ExperimentConfig().arm, [100]
            channels = ArmChannels(mp.get_context("spawn"), run_signal=state.run_signal,
                                   stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
                                   signal_lock=state.motion_lock)
            channels.health_until.value = 1000
            driver = Mock(mode=1)
            driver.servo.return_value = 0
            worker = ArmWorker(config, channels, None, driver, clock_ns=lambda: now[0])
            worker.status["valid"] = True
            worker.status["prepared_run_id"] = state.run_id
            worker.status["frame"]["timestamp_ns"] = 100
            worker.status["frame"]["qpos"] = config.home_qpos
            goal = np.zeros((), GOAL_DTYPE)
            goal[()] = (1, state.run_id, -1, 200, 300, config.home_qpos)
            worker.active = goal
            def sample(stamp):
                if change == "revoke":
                    state.run_signal.value += 1
                if change == "expire":
                    now[0] = 300
                return np.asarray(config.home_qpos), np.zeros(7), np.zeros(7)
            worker.reference = SimpleNamespace(sample=sample)
            if change == "expire":
                with self.assertRaisesRegex(RuntimeError, "input_decision_expired"):
                    worker.tick()
            else:
                worker.tick()
            self.assertEqual(driver.servo.call_count, int(change == "none"))
            if change == "revoke":
                self.assertTrue(channels.stop.is_set())

    def test_step_last_row_at_cancellation_submission_and_wait(self):
        for stage in ("wait", "mapping_cancel", "mapping_expired", "submit_cancel", "sdk_fault",
                      "late_return", "rejected", "ik_fault"):
            with self.subTest(stage=stage):
                row, events, now = observation_row(), [], [1_000_000_000]
                recorder = Mock(start_attempted=True, staging_path=None, written_frames=1)
                recorder.save_episode.return_value = Path("synthetic")
                episode = running_episode(recorder=recorder)
                if stage == "mapping_expired":
                    row.hand["timestamp_ns"] = now[0] - int(episode.runtime.hand.feedback_max_age_s * 1e9) + 3
                decoded = SimpleNamespace(arm_qpos=row.arm["qpos"][0], hand_qpos=row.hand["qpos"][0],
                                          ik_result=None, rejection_reason="synthetic rejection")
                def mapped(*args):
                    events.append("map")
                    if stage == "mapping_cancel":
                        episode.runtime_state.request_stop()
                    if stage == "mapping_expired":
                        now[0] += 4
                    if stage == "rejected":
                        return SimpleNamespace(**{**vars(decoded), "arm_qpos": None})
                    if stage == "ik_fault":
                        return SimpleNamespace(arm_qpos=None, ik_result=SimpleNamespace(
                            failure_kind=IKFailureKind.INVALID_OUTPUT, reason="synthetic nonfinite IK"))
                    return decoded
                def submitted(*args, **kwargs):
                    events.append("submit")
                    if stage == "submit_cancel":
                        raise OperationCancelled("synthetic cancellation")
                    if stage == "sdk_fault":
                        raise RobotError("synthetic partial send")
                    if stage == "late_return":
                        now[0] = episode.next_step_ns
                episode.action_mapper.map.side_effect = mapped
                episode.robot.send_action.side_effect = submitted
                episode.robot.stop.side_effect = lambda: events.append("stop")
                recorder.add_frame.side_effect = lambda *args: events.append("record")
                action = np.concatenate((decoded.arm_qpos, decoded.hand_qpos))
                if stage != "wait":
                    episode.plan = ActionPlan(np.tile(action, (4, 1)), 0, 4, now[0] + 100_000_000, {})
                with patch("dexmani_real.deployment.episode.read_observation", return_value=row), \
                     patch("time.monotonic_ns", side_effect=lambda: now[0]):
                    if stage in {"sdk_fault", "ik_fault"}:
                        with self.assertRaises(RuntimeError):
                            episode.step()
                    else:
                        episode.step()
                recorder.add_frame.assert_called_once()
                command = recorder.add_frame.call_args.args[1]
                self.assertEqual(command is not None, stage in {"submit_cancel", "sdk_fault", "late_return"})
                self.assertIsNone(episode.pending_frame)
                if stage == "wait":
                    self.assertIsNone(episode.termination)
                    self.assertEqual(events, ["record"])
                else:
                    self.assertIsNotNone(episode.termination)
                    self.assertFalse(episode.has_motion_authority())
                    self.assertLess(events.index("stop"), events.index("record"))
                    recorder.save_episode.assert_called_once()

    def test_rtc_prefix_freezes_only_after_query_submission(self):
        episode, row = running_episode(), observation_row()
        episode.execution = replace(episode.execution, execution_mode="rtc", prefetch_steps=2,
                                    rtc_guidance_cap=0.)
        action = np.concatenate((row.arm["qpos"][0], row.hand["qpos"][0]))
        actions = np.tile(action, (8, 1))
        episode.plan = ActionPlan(actions, 0, 4, 1_500_000_000, {})
        episode.action_mapper.map.return_value = SimpleNamespace(arm_qpos=action[:7], hand_qpos=action[7:])
        episode.action_mapper.control_from_command.side_effect = lambda command, mode: np.concatenate(
            (command.arm_qpos, command.hand_qpos))
        episode.action_mapper.frozen_rejection.return_value = None
        with patch("time.monotonic_ns", return_value=1_000_000_000):
            prefix, commands, issue = episode.prepare_prefix(row, 2)
            self.assertIsNone(issue)
            self.assertEqual(set(commands), {2, 3})
            self.assertEqual(episode.plan.frozen, {})
            self.assertFalse(np.shares_memory(prefix, actions))
            projected = project_policy_row(row, episode.policy_info.observation_fields, episode.dependencies)
            self.assertIsNone(episode.submit((projected, projected), 2, handoff_slot=4,
                                            prefix=prefix, commands=commands))
        np.testing.assert_array_equal(actions, np.tile(action, (8, 1)))
        episode.robot.send_action.assert_not_called()
        self.assertEqual(episode.model.submit.call_args.kwargs["delay_steps"], 2)
        self.assertEqual(episode.model.submit.call_args.kwargs["rtc_guidance_cap"], 0.)
        self.assertIs(episode.command_for_slot(row, 2)[0], commands[2])
        episode.query.result = actions
        plan, issue = handoff_plan(episode.query, run_id=episode.run_id, slot=4, mode="rtc",
                                  action_steps=4, now_ns=1_000_000_000, authorized=True)
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (2, 8))

    def test_robot_close_attempts_remaining_resources_after_error(self):
        state, runtime, events = RuntimeState(), ExperimentConfig(), []
        robot = DexManiRobot(runtime.arm, runtime.hand, run_signal=state.run_signal,
                             stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
                             signal_lock=state.motion_lock)
        robot.owner = threading.get_ident()
        robot.arm, robot.hand = Mock(), Mock(is_connected=True)
        def failed_close():
            events.append("arm_close")
            raise RobotError("synthetic close failure")
        robot.arm.close.side_effect = failed_close
        robot.hand.disconnect.side_effect = lambda: events.append("hand_disconnect")
        with self.assertRaises(RobotError):
            robot.close()
        self.assertEqual(events, ["arm_close", "hand_disconnect"])
        self.assertTrue(robot.motion_active)

    def test_retired_reset_failure_is_visible_and_cannot_admit(self):
        episode = running_episode()
        epoch = episode.run_id
        episode.preparing_epoch = epoch
        episode.run_id = None
        episode.termination = object()
        episode.model.poll = lambda: ("reset_episode", ModelResult(
            None, ValueError("synthetic reset failure"), 1, 2, dict(preparing_epoch=epoch),
        ))
        episode.admit = Mock()
        with self.assertRaisesRegex(RuntimeError, "retired model reset_episode failed"):
            episode.poll_model()
        episode.admit.assert_not_called()
        self.assertTrue(episode.runtime_state.error_state)
        self.assertIsNone(episode.preparing_epoch)
        episode.results.record_errors.assert_called_once()

    def test_worker_reset_predict_close_and_retired_failure(self):
        events = []
        model = SimpleNamespace(
            warmup=lambda **kwargs: events.append("load"),
            reset_episode=lambda: events.append("reset"),
            predict=Mock(side_effect=ValueError("synthetic prediction failure")),
            close=lambda: events.append("close"),
        )
        worker = InferenceWorker(lambda: model, build_observation=lambda rows: rows,
                                 reset_observation=lambda: events.append("clear"))
        self.addCleanup(worker.close)
        for operation in ("load", "reset_episode"):
            worker.submit(operation)
            deadline = time.monotonic() + 2
            while (result := worker.poll()) is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.001)
            self.assertIsNone(result[1].error)
        worker.submit("predict", (), query_context=dict(run_id=3, query_id=7))
        worker.close()
        with self.assertRaises(RuntimeError):
            worker.submit("reset_episode")
        result = worker.close_result(timeout_s=2)
        self.assertEqual(events, ["load", "clear", "reset", "close"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retired_task"]["query_context"], dict(run_id=3, query_id=7))
        self.assertEqual(result["retired_task"]["exception_type"], "ValueError")

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
        deadline = time.monotonic() + 2
        while (completion := worker.poll()) is None:
            self.assertTrue(worker.has_pending_task)
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.001)
        self.assertIsNone(completion[1].error)
        self.assertFalse(worker.has_pending_task)
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
                record_needs=ObservationNeeds(), dependencies=frozenset(), results_dir=root, run_config={},
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

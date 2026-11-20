"""验证在线 HOI 逐行观测、原始录制和故障恢复。"""

import unittest
from concurrent.futures import Future
from multiprocessing import get_context
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.config.real import RealConfig
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.data.raw import RawReader, camera_metadata
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.robot.commands import FeedbackUnavailable
from dexmani_real.robot.system import RobotState
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import (
    check_hoi_status,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import SessionState

MS = 1_000_000
REQUEST_NS = 1000 * MS
SAMPLE_NS = 1400 * MS


class OnlineObservationTest(unittest.TestCase):
    def setUp(self):
        self.runtime = RealConfig(
            execution=ExecutionConfig(control_dt_s=0.02, action_steps=1),
            camera=CameraConfig(width=2, height=2),
        )
        color = CameraIntrinsics(2, 2, 2.0, 2.0, 0.5, 0.5, "none", (0.0,) * 5)
        self.camera = CameraSnapshot(None, color, 0.001, np.eye(4))
        self.channels = SensorChannels(
            get_context("spawn"), camera=self.runtime.camera,
            pointcloud_num_points=4, include_derived_rgb=True, hoi=True,
        )
        self.camera_ring = cast(SharedRing, self.channels.camera_ring)
        self.hoi_ring = cast(SharedRing, self.channels.hoi_ring)
        self.arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        self.hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        self.arm["timestamp_ns"] = 1370 * MS
        self.hand["timestamp_ns"] = 1380 * MS
        self.arm["qpos"] = np.linspace(0.01, 0.07, self.arm["qpos"].size)
        self.hand["qpos"] = np.linspace(0.10, 0.21, self.hand["qpos"].size)
        self.robot = SimpleNamespace(
            read_state=Mock(return_value=RobotState(self.arm, self.hand)),
            hand_config=self.runtime.hand,
            hand_feedback_pending=False,
            hold_hand=Mock(return_value=True),
            send_action=Mock(return_value=SAMPLE_NS),
        )
        self.raw_rgb = np.full((2, 2, 3), 32, dtype=np.uint8)
        self.derived_rgb = np.full((2, 2, 3), 64, dtype=np.uint8)
        self.depth = np.full((2, 2), 600, dtype=np.uint16)
        self.object_cloud = np.full((HOI_NUM_POINTS, 3), 0.25, dtype=np.float32)
        clock_patch = patch("time.monotonic_ns", return_value=SAMPLE_NS)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def make_observations(self, fields):
        builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), self.runtime)
        builder.configure_camera(self.camera)
        return builder

    def make_rollout(self, fields):
        state = SessionState()
        rollout = Rollout(
            state, self.runtime,
            SimpleNamespace(n_obs_steps=2, future_steps=3, requires_prev_action=False, action_mode="joint"),
            episode_cancel_epoch=state.cancel_epoch, robot=self.robot,
            observations=self.make_observations(fields),
            sensors=SimpleNamespace(channels=self.channels),
            action_mapper=SimpleNamespace(map_joint=lambda arm, hand: SimpleNamespace(arm_qpos=arm, hand_qpos=hand)),
            model=Mock(future=None, poll=Mock(return_value=None)), max_duration_s=None,
            index=7, output_dir=None, record_raw=True, effective_parameters={},
        )
        self.clock.return_value = REQUEST_NS
        rollout._request_hoi(rollout.result.index)
        self.clock.return_value = SAMPLE_NS
        return rollout

    def write_camera(self, source_ns, rgb):
        return self.camera_ring.write(timestamp_ns=source_ns, rgb=rgb, depth=self.depth)

    def write_hoi(self, source_ns, sequence, *, valid=True):
        self.hoi_ring.write(
            episode_id=7, timestamp_ns=source_ns, source_camera_sequence=sequence,
            valid=valid, object_pointcloud=self.object_cloud, rgb=self.derived_rgb,
        )

    def joint_state(self):
        return np.concatenate((self.arm["qpos"][0], self.hand["qpos"][0]), dtype=np.float32)

    def test_hoi_projection_and_raw_recording_use_same_robot_sample(self):
        old_sequence = self.write_camera(990 * MS, self.raw_rgb)
        rollout = self.make_rollout(("joint_state", "object_pointcloud", "rgb"))
        self.assertEqual(check_hoi_status(self.channels, self.runtime), ["hoi_unavailable"])

        hoi_sequence = self.write_camera(1350 * MS, self.derived_rgb)
        for source_ns, sequence in ((REQUEST_NS, hoi_sequence), (1350 * MS, old_sequence)):
            self.write_hoi(source_ns, sequence)
            self.assertEqual(check_hoi_status(self.channels, self.runtime), ["hoi_unavailable"])
        self.write_hoi(1350 * MS, hoi_sequence)
        raw_sequence = self.write_camera(1390 * MS, self.raw_rgb)

        row, raw = rollout.sample_observation()
        self.robot.read_state.assert_called_once_with()
        self.assertEqual(row.hoi["source_camera_sequence"], hoi_sequence)
        self.assertEqual(row.camera["timestamp_ns"], 1350 * MS)
        self.assertEqual(row.camera["ring_sequence"], hoi_sequence)
        self.assertEqual(raw.camera["ring_sequence"], raw_sequence)
        self.assertIsNone(raw.hoi)
        np.testing.assert_array_equal(raw.camera["rgb"], self.raw_rgb)
        np.testing.assert_array_equal(raw.camera["depth"], self.depth)
        self.assertTrue(rollout.update_history(row))
        policy = rollout.observations.build(rollout.history)
        np.testing.assert_array_equal(policy["joint_state"], np.repeat(self.joint_state()[None], 2, axis=0))
        np.testing.assert_array_equal(policy["object_pointcloud"][0], self.object_cloud)
        np.testing.assert_array_equal(policy["rgb"][0], self.derived_rgb)

        with TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, control_hz=50, read_xhand_crc=dict)
            recorder.start(
                task_label="hoi_snapshot", collection_source="policy_rollout",
                metadata=camera_metadata(self.camera),
            )
            try:
                self.assertIsNone(recorder.append(raw))
            finally:
                recorder.finish(save=True, reason="completed")
            recorder.check_error()
            published_path = recorder.published_path
            assert published_path is not None
            with RawReader(published_path) as reader:
                timing = reader.read_timing(source_clocks=("camera_ns", "arm_ns", "hand_ns"))[0]
                self.assertEqual(timing["camera_ns"], 1390 * MS)
                self.assertEqual(timing["camera_sequence"], raw_sequence)
                self.assertEqual(timing["arm_ns"], 1370 * MS)
                self.assertEqual(timing["hand_ns"], 1380 * MS)
                np.testing.assert_allclose(reader["arm_qpos"][0], self.arm["qpos"][0])
                np.testing.assert_allclose(reader["hand_qpos"][0], self.hand["qpos"][0])
                np.testing.assert_array_equal(reader["depth"][0], self.depth)
                np.testing.assert_allclose(list(reader.iter_rgb_frames()), self.raw_rgb[None], atol=2)

        rollout.stop("finished")
        self.assertEqual(check_hoi_status(self.channels, self.runtime), ["hoi_unavailable"])

    def test_unavailable_invalid_and_stale_hoi_hold_then_resume(self):
        for issue, source_ns, valid in (
            ("hoi_unavailable", None, False),
            ("hoi_pose_invalid", 1350 * MS, False),
            ("hoi_stale", 1100 * MS, True),
        ):
            with self.subTest(issue=issue):
                rollout = self.make_rollout(("joint_state", "object_pointcloud"))
                if source_ns is not None:
                    self.write_hoi(source_ns, self.write_camera(source_ns, self.derived_rgb), valid=valid)
                self.write_camera(1390 * MS, self.raw_rgb)
                self.assertEqual(check_hoi_status(self.channels, self.runtime), [issue])
                rollout.started_ns = SAMPLE_NS
                rollout.recorder = Mock(append=Mock(return_value=None))
                self.robot.send_action.reset_mock()

                rollout.step()
                self.assertEqual(rollout.state, "HOI_HOLD")
                self.assertTrue(self.robot.send_action.call_args.args[0].arm_hold)
                rollout.recorder.mark_gap.assert_called_once_with()

                self.write_hoi(1395 * MS, self.write_camera(1395 * MS, self.derived_rgb))
                self.write_camera(1398 * MS, self.raw_rgb)
                with patch.object(self.hoi_ring, "read", wraps=self.hoi_ring.read) as read_hoi:
                    rollout.step()
                read_hoi.assert_called_once()
                self.assertEqual(rollout.state, "RESUMING")
                self.clock.return_value += rollout.dt_ns
                rollout.step()
                self.assertEqual(rollout.state, "RESUMING")
                rollout.recorder.append.assert_not_called()
                _, prediction_kwargs = rollout.model.predict.call_args
                self.assertTrue(prediction_kwargs["reset_episode"])

                future: Future[np.ndarray] = Future()
                future.set_result(self.joint_state()[None])
                rollout.model.poll.return_value = future
                self.clock.return_value += rollout.dt_ns
                rollout.step()
                self.assertEqual(rollout.state, "RUNNING")
                rollout.recorder.append.assert_called_once()
                recorded, command = rollout.recorder.append.call_args.args
                self.assertIs(command, self.robot.send_action.call_args.args[0])
                self.assertFalse(command.arm_hold)
                self.assertIsNone(recorded.hoi)
                np.testing.assert_array_equal(recorded.arm["qpos"], self.arm["qpos"])
                np.testing.assert_array_equal(recorded.hand["qpos"], self.hand["qpos"])

    def test_fresh_hoi_does_not_mask_stale_device_feedback(self):
        rollout = self.make_rollout(("joint_state", "object_pointcloud"))
        self.write_hoi(1350 * MS, self.write_camera(1350 * MS, self.derived_rgb))
        self.assertEqual(check_hoi_status(self.channels, self.runtime), [])
        self.arm["timestamp_ns"] = 1100 * MS
        self.hand["timestamp_ns"] = 1110 * MS
        with self.assertRaisesRegex(
            FeedbackUnavailable, "arm_feedback_stale; hand_timestamp_ns_unavailable_or_stale",
        ):
            rollout.sample_observation()

    def test_ordinary_rgb_scene_cloud_and_joint_policies(self):
        rgb_needs = self.make_observations(("rgb",)).needs
        row = read_observation(self.channels, self.robot, needs=rgb_needs)
        self.assertEqual(check_observation(row, self.runtime, rgb_needs), ["camera_source_unavailable"])

        scene_ns = 1350 * MS
        scene_sequence = self.write_camera(scene_ns, self.derived_rgb)
        scene = np.full((4, 6), 0.5, dtype=np.float32)
        pointcloud_ring = cast(SharedRing, self.channels.pointcloud_ring)
        pointcloud_ring.write(
            timestamp_ns=scene_ns, source_camera_sequence=scene_sequence,
            point_cloud=scene, rgb=self.derived_rgb,
        )
        camera_sequence = self.write_camera(1390 * MS, self.raw_rgb)
        expected_policies: tuple[dict[str, np.ndarray], ...] = (
            {"rgb": self.raw_rgb},
            {"point_cloud": scene, "rgb": self.derived_rgb},
            {"joint_state": self.joint_state()},
        )
        for expected in expected_policies:
            fields = tuple(expected)
            with self.subTest(fields=fields):
                builder = self.make_observations(fields)
                row = read_observation(self.channels, self.robot, needs=builder.needs)
                batch = builder.build([builder.project(row)])
                self.assertEqual(tuple(batch), fields)
                for name, value in expected.items():
                    np.testing.assert_array_equal(batch[name][0], value)
                if "rgb" in fields:
                    sequence = scene_sequence if "point_cloud" in fields else camera_sequence
                    assert row.camera is not None
                    self.assertEqual(row.camera["ring_sequence"], sequence)

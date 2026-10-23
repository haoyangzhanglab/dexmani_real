import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fakes import FakeClock, observation
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.observation import build_policy_observation, policy_sources
from dexmani_real.ipc.schema import make_camera_frame_dtype, make_pointcloud_frame_dtype
from dexmani_real.runtime.observation import ObservationNeeds, read_observation, upstream_deadline_ns


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.runtime = ExperimentConfig()
        self.time_patch = patch("time.monotonic_ns", self.clock.monotonic_ns)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.row = observation(self.clock)
        self.robot = SimpleNamespace(read_state=lambda: self.row)
        self.camera = np.zeros(1, make_camera_frame_dtype((4, 4, 3), (4, 4)))
        self.camera["header"]["timestamp_ns"] = self.clock.now
        self.cloud = np.zeros(1, make_pointcloud_frame_dtype(4))
        self.cloud["timestamp_ns"] = self.clock.now
        self.cloud["source_camera_sequence"] = 17
        self.channels = SimpleNamespace(camera_ring=Mock(), pointcloud_ring=Mock())
        self.channels.camera_ring.read_latest.return_value = self.camera, self.clock.now, 18
        self.channels.camera_ring.read_sequence.return_value = self.camera, self.clock.now, 17
        self.channels.camera_ring.read_latest_field.return_value = self.camera["header"], self.clock.now, 18
        self.channels.pointcloud_ring.read_latest.return_value = self.cloud, self.clock.now, 3

    def test_policy_needs_matrix(self):
        cases = [
            (("joint_state",), False, ("none", False, (), False)),
            (("finger_tactile_force",), False, ("none", False, ("aggregate",), False)),
            (("taxel_tactile_force",), False, ("none", False, ("dense",), False)),
            (("camera_intrinsic",), False, ("header", False, (), True)),
            (("camera_extrinsic",), False, ("header", False, (), True)),
            (("point_cloud",), False, ("none", True, (), True)),
            (("rgb",), False, ("rgbd", False, (), True)),
            (("rgb", "point_cloud"), False, ("rgbd", True, (), True)),
            (("joint_state",), True, ("rgbd", False, ("aggregate", "dense"), True)),
        ]
        for fields, raw, expected in cases:
            with self.subTest(fields=fields, raw=raw):
                needs = ObservationNeeds.for_policy(fields, record_raw=raw)
                self.assertEqual((needs.camera, needs.point_cloud, needs.required_tactile, needs.camera_worker), expected)
                self.assertFalse(needs.vr)

    def test_intrinsics_extrinsics_only_copy_header(self):
        needs = ObservationNeeds.for_policy(("camera_intrinsic", "camera_extrinsic"), record_raw=False)
        row = read_observation(self.channels, self.runtime, self.robot, needs=needs)
        self.assertNotIn("rgb", row.camera)
        self.channels.camera_ring.read_latest.assert_not_called()
        self.channels.camera_ring.read_latest_field.assert_called_once_with("header")

    def test_cloud_only_does_not_copy_camera(self):
        row = read_observation(self.channels, self.runtime, self.robot,
                               needs=ObservationNeeds.for_policy(("point_cloud",), record_raw=False))
        self.assertIsNone(row.camera)
        self.assertEqual(row.pointcloud_camera_sequence, 17)
        self.assertEqual(self.channels.camera_ring.mock_calls, [])

    def test_rgb_cloud_pair_uses_source_identity(self):
        needs = ObservationNeeds.for_policy(("rgb", "point_cloud"), record_raw=False)
        row = read_observation(self.channels, self.runtime, self.robot, needs=needs)
        self.assertEqual(row.camera["ring_sequence"], row.pointcloud_camera_sequence)
        self.channels.camera_ring.read_sequence.assert_called_once_with(17)
        self.channels.camera_ring.read_latest.assert_not_called()
        self.channels.camera_ring.read_sequence.return_value = None
        self.assertIsNone(read_observation(self.channels, self.runtime, self.robot, needs=needs))

    def test_fresh_processing_cannot_refresh_old_camera_source(self):
        self.cloud["timestamp_ns"] = 1
        self.cloud["processing_started_ns"] = self.clock.now
        issues = []
        row = read_observation(self.channels, self.runtime, self.robot, issues=issues,
                               needs=ObservationNeeds.for_policy(("point_cloud",), record_raw=False))
        self.assertIsNone(row)
        self.assertIn("pointcloud_stale", issues)

    def test_tactile_freshness_is_per_consumed_modality(self):
        self.row.hand["tactile_dense_timestamp_ns"] = 1
        aggregate = ObservationNeeds.for_policy(("finger_tactile_force",), record_raw=False)
        self.assertIsNotNone(read_observation(None, self.runtime, self.robot, needs=aggregate))
        dense = ObservationNeeds.for_policy(("taxel_tactile_force",), record_raw=False)
        self.assertIsNone(read_observation(None, self.runtime, self.robot, needs=dense))
        raw = ObservationNeeds.for_policy(("joint_state",), record_raw=True)
        self.assertIsNone(read_observation(self.channels, self.runtime, self.robot, needs=raw))

    def test_conversion_preserves_source_times_and_values(self):
        row = replace(observation(self.clock, camera=True), pointcloud_timestamp_ns=self.clock.now - 100)
        info = SimpleNamespace(observation_fields=("joint_state", "finger_tactile_force", "rgb", "camera_extrinsic"))
        extrinsic = np.arange(12, dtype=np.float32)
        arrays = build_policy_observation((row,), info, camera_parameters={"camera_extrinsic": extrinsic})
        np.testing.assert_array_equal(arrays["finger_tactile_force"], row.hand["tactile_aggregate"])
        np.testing.assert_array_equal(arrays["camera_extrinsic"][0], extrinsic)
        self.assertEqual(policy_sources(row, info.observation_fields)["camera_extrinsic"], (self.clock.now,))
        self.assertEqual(upstream_deadline_ns(row, self.runtime),
                         self.clock.now - 100 + int(self.runtime.camera.max_frame_age_s * 1e9))

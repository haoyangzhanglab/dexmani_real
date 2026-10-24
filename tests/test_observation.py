"""来源时钟、独立准入和模型历史；只使用合成数据及离线 FK。"""

import gc
import multiprocessing as mp
import tempfile
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.data.results import EpisodeResults
from dexmani_real.deployment.inference import InferenceWorker
from dexmani_real.deployment.observation import (
    build_policy_observation, policy_needs,
    policy_source_times, project_policy_row,
)
from dexmani_real.deployment.session import build_observation_kinematics
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.observation import (
    ObservationNeeds, control_deadline_ns, observation_issues, read_observation,
)
from tests.offline_fixtures import make_row


class SourceTimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = ExperimentConfig()
        self.feedback = XHandFeedback(self.runtime.hand)
        self.feedback.reset(1_000_000_000)
        self.sample = SimpleNamespace(
            qpos=np.ones(12), tactile_aggregate=np.ones((5, 3)),
            tactile_dense=np.ones((5, 120, 3)), commboard_err=np.zeros(12),
            jointboard_err=np.zeros(12), tipboard_err=np.zeros(12),
        )

    def test_slow_query_uses_start_as_source_and_completion_as_age(self):
        source = 1_000_000_000
        self.feedback.consume(self.sample, source_ns=source, now_ns=source + 1)
        budget = int(self.runtime.hand.feedback_max_age_s * 1e9)
        with self.assertRaisesRegex(RuntimeError, "age"):
            self.feedback.consume(self.sample, source_ns=source + 10, now_ns=source + 10 + budget)
        self.assertEqual(self.feedback.latest["timestamp_ns"][0], source + 10)

    def test_crc_and_partial_vectors_keep_original_clocks(self):
        source = 1_000_000_000
        first = self.feedback.consume(self.sample, source_ns=source, now_ns=source + 1)
        held = self.feedback.consume(None, source_ns=source + 10, now_ns=source + 11)
        self.assertIs(first, held)
        self.sample.tactile_dense[:] = 7
        self.sample.tactile_dense[0, 0, 1] = np.nan
        fresh = self.feedback.consume(self.sample, source_ns=source + 20, now_ns=source + 21)
        np.testing.assert_array_equal(fresh["tactile_dense"][0, 0, 0], [1, 1, 1])
        self.assertEqual(fresh["tactile_dense_timestamp_ns"][0], source)
        self.assertEqual(fresh["tactile_aggregate_timestamp_ns"][0], source + 20)
        row = make_row(source + int(self.runtime.hand.feedback_max_age_s * 1e9))
        row.hand = fresh
        control = ObservationNeeds()
        record = ObservationNeeds(required_tactile=("dense",))
        self.assertEqual(observation_issues(row, self.runtime, control, now_ns=row.observation_timestamp_ns), [])
        self.assertIn("dense", observation_issues(row, self.runtime, record, now_ns=row.observation_timestamp_ns)[0])

    def test_initial_deadline_does_not_restart_on_missing_feedback(self):
        self.feedback.reset(1_000_000_000, deadline_ns=1_000_000_100)
        self.assertIsNone(self.feedback.consume(None, source_ns=1_000_000_001, now_ns=1_000_000_099))
        with self.assertRaisesRegex(RuntimeError, "initial feedback timed out"):
            self.feedback.consume(None, source_ns=1_000_000_099, now_ns=1_000_000_100)

    def test_inference_metrics_keep_twenty_ten_thirty_boundaries(self):
        worker = InferenceWorker(lambda: None, build_observation=lambda rows: rows)
        worker.model = SimpleNamespace(predict=lambda values: values)
        with patch("dexmani_real.deployment.inference.time.monotonic_ns", side_effect=[1, 20_000_001, 30_000_001]):
            result = worker.run_operation("predict", ([1],), {})
        worker.executor.shutdown()
        with tempfile.TemporaryDirectory() as directory:
            results = EpisodeResults(directory, "offline")
            results.begin_episode()
            results.entered(1)
            results.record_inference(result)
            metrics = results.current["metrics"]
        self.assertAlmostEqual(metrics["observation_build_mean_s"], .02)
        self.assertAlmostEqual(metrics["inference_mean_s"], .01)
        self.assertAlmostEqual(metrics["prediction_total_mean_s"], .03)


class ObservationProjectionTests(unittest.TestCase):
    def test_unsupported_fields_fail_at_requirement_resolution(self):
        with self.assertRaisesRegex(ValueError, "unsupported_sensor"):
            policy_needs(("joint_state", "unsupported_sensor"))

    def test_recording_union_does_not_change_control_deadline_or_sample_twice(self):
        runtime = ExperimentConfig()
        physical = make_row()
        control = policy_needs(("joint_state",))
        record = ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
        robot = SimpleNamespace(read_state=Mock(return_value=physical))
        channels = SimpleNamespace(camera_ring=SimpleNamespace(read_fields=Mock(return_value=None)))
        with patch("dexmani_real.runtime.observation.time.monotonic_ns", return_value=1_000_000_001):
            row = read_observation(channels, robot, needs=control.union(record))
            self.assertEqual(observation_issues(row, runtime, control), [])
            self.assertIn("camera_source_unavailable_or_unmatched", observation_issues(row, runtime, record))
            deadline = control_deadline_ns(row, runtime, control)
            unrecorded = read_observation(None, robot, needs=control)
            self.assertEqual(control_deadline_ns(unrecorded, runtime, control), deadline)
        self.assertEqual(robot.read_state.call_count, 2)
        channels.camera_ring.read_fields.assert_called_once()
        self.assertEqual(control.required_tactile, ())
        projection = project_policy_row(row, ("joint_state",))
        self.assertEqual(set(projection.values), {"arm_qpos", "hand_qpos"})
        self.assertFalse(np.shares_memory(projection.values["hand_qpos"], row.hand))
        self.assertFalse(projection.values["hand_qpos"].flags.writeable)

    def test_cloud_and_recording_image_use_exact_sequence(self):
        runtime = ExperimentConfig()
        source = make_row()
        cloud = np.zeros(1, dtype=[("source_camera_sequence", "i8"), ("timestamp_ns", "i8"), ("point_cloud", "f4", (2, 6))])
        cloud["source_camera_sequence"] = 7
        cloud["timestamp_ns"] = 1_000_000_000
        camera_ring = SimpleNamespace(read_fields=Mock(return_value=None))
        channels = SimpleNamespace(
            camera_ring=camera_ring,
            pointcloud_ring=SimpleNamespace(read_latest=Mock(return_value=(cloud, 1, 9))),
        )
        control = policy_needs(("point_cloud",))
        record = ObservationNeeds(camera="rgbd")
        with patch("dexmani_real.runtime.observation.time.monotonic_ns", return_value=1_000_000_010):
            row = read_observation(channels, SimpleNamespace(read_state=lambda: source), needs=control.union(record))
            self.assertEqual(observation_issues(row, runtime, control), [])
            self.assertTrue(observation_issues(row, runtime, record))
        camera_ring.read_fields.assert_called_once_with(("header", "rgb", "depth"), sequence=7)
        self.assertEqual(row.camera_sequence, 7)
        self.assertEqual(row.pointcloud_timestamp_ns, 1_000_000_000)

    def test_static_parameters_use_control_anchor_and_no_dynamic_camera(self):
        row = make_row()
        row.camera = None
        fields = ("camera_intrinsic", "camera_extrinsic")
        needs = policy_needs(fields)
        self.assertTrue(needs.camera_worker)
        self.assertEqual(needs.camera, "none")
        self.assertEqual(policy_source_times(row, fields), (1_000_000_000, 1_000_000_000))
        self.assertEqual(observation_issues(row, ExperimentConfig(), needs, now_ns=1_000_000_001), [])
        parameters = {"camera_intrinsic": np.eye(3).ravel(), "camera_extrinsic": np.eye(4)[:3].ravel()}
        result = build_policy_observation(
            [project_policy_row(row, fields)], SimpleNamespace(observation_fields=fields),
            camera_parameters=parameters,
        )
        self.assertEqual(result["camera_extrinsic"].shape, (1, 12))

    def test_field_copies_do_not_retain_recording_depth_and_repeated_frames_keep_time(self):
        dtype = np.dtype([("header", [("timestamp_ns", "i8")]), ("rgb", "u1", (2, 2, 3)), ("depth", "u2", (2, 2))])
        ring = SharedMemoryRingBuffer(mp.get_context("spawn"), dtype, 2)
        header = np.zeros((), dtype=dtype["header"])
        header["timestamp_ns"] = 1_000_000_000
        ring.write(header=header, rgb=np.zeros((2, 2, 3), dtype="u1"), depth=np.ones((2, 2), dtype="u2"))
        channels = SimpleNamespace(camera_ring=ring)
        row = make_row()
        with patch("dexmani_real.runtime.observation.time.monotonic_ns", return_value=1_000_000_010):
            robot = SimpleNamespace(read_state=lambda: row)
            sampled = read_observation(channels, robot, needs=ObservationNeeds(camera="rgbd"))
            again = read_observation(channels, robot, needs=ObservationNeeds(camera="rgbd"))
        self.assertEqual(sampled.camera["timestamp_ns"], again.camera["timestamp_ns"])
        self.assertEqual(sampled.camera_sequence, again.camera_sequence)
        depth = weakref.ref(sampled.camera["depth"])
        projected = project_policy_row(sampled, ("rgb",))
        rgb = sampled.camera["rgb"]
        del sampled
        gc.collect()
        self.assertIsNone(depth())
        self.assertIs(projected.values["rgb"], rgb)

    def test_fresh_but_offset_sources_are_not_hard_synchronized(self):
        row = make_row()
        row.camera["timestamp_ns"] -= 20_000_000
        self.assertEqual(observation_issues(row, ExperimentConfig(), ObservationNeeds(camera="rgbd"), now_ns=1_000_000_001), [])


class GeometryHistoryTests(unittest.TestCase):
    def test_overlap_reuses_each_physical_row_once_and_reset_recomputes(self):
        runtime = ExperimentConfig()
        fields = ("eef_pose", "fingertip_xyz", "hand_pointcloud")
        info = SimpleNamespace(
            observation_fields=fields, n_obs_steps=4,
            fingertip_link_names=runtime.hand.fingertip_link_names,
        )
        execution = replace(runtime.execution, execution_mode="async", action_steps=1, prefetch_steps=1)
        cached = build_observation_kinematics(info, replace(runtime, execution=execution))
        direct = replace(cached, row_features=None)
        rows = []
        for index in range(6):
            row = make_row(1_000_000_000 + index)
            row.arm["qpos"] = np.asarray(runtime.arm.home_qpos) + index * .001
            row.hand["qpos"] = np.deg2rad(runtime.hand.home_qpos_deg) + index * .001
            rows.append(project_policy_row(row, fields))
        with patch.object(cached.arm_fk, "compute", wraps=cached.arm_fk.compute) as arm, patch.object(
            cached.hand_fk, "compute_frame_placements", wraps=cached.hand_fk.compute_frame_placements,
        ) as hand:
            expected = [build_policy_observation(rows[i:i + 4], info, kinematics=direct) for i in range(3)]
            self.assertEqual(arm.call_count, 12)
            self.assertEqual(hand.call_count, 12)
            arm.reset_mock()
            hand.reset_mock()
            for index in range(3):
                actual = build_policy_observation(rows[index:index + 4], info, kinematics=cached)
                for field in fields:
                    np.testing.assert_allclose(actual[field], expected[index][field], rtol=1e-6, atol=1e-7)
                self.assertEqual(len(cached.row_features), 4)
            self.assertEqual(arm.call_count, 6)
            self.assertEqual(hand.call_count, 6)
            build_policy_observation(rows[2:], info, kinematics=cached)
            self.assertEqual(arm.call_count, 6)
            cached.row_features.clear()
            build_policy_observation(rows[2:], info, kinematics=cached)
            self.assertEqual(arm.call_count, 10)
        one = SimpleNamespace(observation_fields=("eef_pose",), n_obs_steps=1)
        self.assertIsNone(build_observation_kinematics(one, replace(runtime, execution=execution)).row_features)
        plain = SimpleNamespace(observation_fields=("joint_state",), n_obs_steps=4)
        self.assertIsNone(build_observation_kinematics(plain, replace(runtime, execution=execution)))

    def test_geometry_subsets_repeated_rows_and_no_overlap(self):
        runtime = ExperimentConfig()
        execution = replace(runtime.execution, execution_mode="sync", action_steps=1)
        runtime = replace(runtime, execution=execution)
        row = make_row()
        for fields in (("eef_pose",), ("fingertip_xyz",), ("hand_pointcloud",)):
            info = SimpleNamespace(
                observation_fields=fields, n_obs_steps=4,
                fingertip_link_names=runtime.hand.fingertip_link_names,
            )
            geometry = build_observation_kinematics(info, runtime)
            rows = [project_policy_row(row, fields)] * 4
            with patch.object(geometry.arm_fk, "compute", wraps=geometry.arm_fk.compute) as arm:
                cached = build_policy_observation(rows, info, kinematics=geometry)
                self.assertEqual(arm.call_count, 1)
                self.assertEqual(len(geometry.row_features), 1)
            direct = build_policy_observation(rows, info, kinematics=replace(geometry, row_features=None))
            self.assertEqual(set(cached), set(fields))
            np.testing.assert_allclose(cached[fields[0]], direct[fields[0]], atol=1e-7)
            no_overlap = replace(runtime, execution=replace(execution, action_steps=3))
            self.assertIsNone(build_observation_kinematics(info, no_overlap).row_features)

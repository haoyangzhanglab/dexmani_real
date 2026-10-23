import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from fakes import FakeClock, camera_metadata, execution_snapshot, file_contents, observation
from dexmani_real.data.canonical.export import CanonicalExporter
from dexmani_real.data.canonical.schema import ProcessingConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.data.raw.schema import RAW_FORMAT, encode_raw_row
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.robot.commands import DispatchResult, DispatchStatus, RobotCommand
from dexmani_real.robot.model import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.sensor.pointcloud.config import PointCloudConfig, SAMPLING_METHOD


class DataContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = FakeClock()

    def recorder(self):
        return AsyncEpisodeRecorder(self.root / "raw", control_hz=20)

    def start(self, recorder):
        recorder.start_episode(task_label="test", collection_source="policy_rollout",
                               metadata=camera_metadata(), arm_execution=execution_snapshot(),
                               arm_control_metadata={}, episode_name="episode_001")

    def test_zero_row_capture_is_published_with_termination(self):
        recorder = self.recorder()
        self.start(recorder)
        path = recorder.save_episode(reason="start_cancelled", arm_execution=execution_snapshot())
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.num_frames, 0)
            self.assertEqual(reader.meta["termination_reason"], "start_cancelled")
            self.assertEqual(reader.meta["format"], RAW_FORMAT)
        before = file_contents(path)
        recorder.close()
        self.assertEqual(file_contents(path), before)

    def test_not_ready_writer_keeps_staging(self):
        recorder = self.recorder()
        with patch.object(RawEpisodeWriter, "open", side_effect=OSError("disk unavailable")):
            with self.assertRaises(RecordingError):
                self.start(recorder)
        self.assertFalse(recorder.is_recording)
        self.assertTrue(recorder.staging_path.is_dir())
        self.assertIsNone(recorder.published_path)
        with self.assertRaises(RecordingError):
            recorder.save_episode(reason="preparation_failed", arm_execution=execution_snapshot())

    def test_full_queue_never_retries_or_counts_rejected_frame(self):
        recorder = self.recorder()
        recorder.recording = True
        for _ in range(recorder.queue.maxsize):
            recorder.queue.put_nowait(object())
        with self.assertRaises(RecordingError):
            recorder.add_frame(observation(self.clock, camera=True))
        self.assertEqual(recorder.submitted_frames, 0)
        self.assertEqual(recorder.queue.qsize(), recorder.queue.maxsize)

    def test_write_encode_and_publish_failures_keep_staging(self):
        targets = (
            "dexmani_real.data.raw.writer.RawEpisodeWriter.append_frame",
            "dexmani_real.data.raw.video.VideoEncoder.write_frame",
            "dexmani_real.data.recorder.atomic_publish",
        )
        for index, target in enumerate(targets):
            with self.subTest(target=target):
                recorder = AsyncEpisodeRecorder(self.root / str(index), control_hz=20)
                self.start(recorder)
                with patch(target, side_effect=OSError("injected failure")):
                    recorder.add_frame(observation(self.clock, camera=True))
                    with self.assertRaises(RecordingError):
                        recorder.save_episode(reason="failure", arm_execution=execution_snapshot())
                self.assertTrue(recorder.staging_path.is_dir())
                self.assertFalse(recorder.episode_dir.exists())
                self.assertIsNone(recorder.published_path)
                self.assertFalse(recorder.thread.is_alive())

    def test_unconfirmed_stop_prevents_publication(self):
        recorder = self.recorder()
        self.start(recorder)
        with self.assertRaises(RecordingError):
            recorder.save_episode(reason="stop_failed", arm_execution=execution_snapshot(stopped=False))
        self.assertTrue(recorder.staging_path.is_dir())
        self.assertFalse(recorder.episode_dir.exists())

    def test_raw_missing_values_remain_nan_and_actions_are_submitted_targets(self):
        row = replace(observation(self.clock, camera=True), hand=None)
        command = RobotCommand(3, np.ones(7), np.ones(12))
        values = encode_raw_row(row, command,
                                DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.NOT_CALLED), 0)
        self.assertTrue(np.isnan(values["hand_current"]).all())
        self.assertTrue(np.isnan(values["action_hand_joint_target"]).all())
        np.testing.assert_array_equal(values["action_arm_joint_target"], np.ones(7))
        self.assertFalse(np.array_equal(values["arm_qpos"], values["action_arm_joint_target"]))

    def fixture(self, name, *, invalid=None):
        # Corruptions are constructed in unpublished synthetic staging only.
        staging = self.root / "raw" / (".tmp_" + name)
        staging.mkdir(parents=True)
        metadata = camera_metadata()
        transform = np.array([[0., -1., 0., .1], [1., 0., 0., -.2],
                              [0., 0., 1., .3], [0., 0., 0., 1.]])
        metadata["camera_T_xarm_base_from_color"] = transform.ravel()
        metadata.update(task_label="test", collection_source="teleop")
        writer = RawEpisodeWriter(staging, control_hz=20)
        try:
            writer.open(metadata, execution_snapshot(), {})
            for index in range(2):
                row = observation(self.clock, camera=True)
                row.arm["qpos"] += .002 * index * np.arange(1, 8)
                row.arm["qvel"] = .01 * np.arange(7) + .1 * (index + 1)
                row.arm["effort"] = .02 * np.arange(7) + .3 * (index + 1)
                row.hand["qpos"] += .01 * np.arange(1, 13) * (index + 1)
                row.hand["current"] = np.arange(12) + 3 * (index + 1)
                row.hand["tactile_aggregate"] = np.arange(15).reshape(5, 3) * .01 + 1.25 + index
                row.hand["tactile_dense"] = np.arange(1800).reshape(5, 120, 3) * .001 + 2.5 + index
                pixels = np.arange(16).reshape(4, 4)
                row.camera["rgb"][:] = np.stack((pixels * 10 + index, pixels * 7 + 20,
                                                 200 - pixels * 9), axis=-1)
                row.camera["depth"][:] = 500 + 20 * index
                command = RobotCommand(3, row.arm["qpos"][0] + .001 * np.arange(1, 8),
                                       row.hand["qpos"][0] + .005 * (index + 1))
                receipt = DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.CRC_UNCONFIRMED,
                                         self.clock.now, (3, index + 1))
                writer.append_frame(row, command, receipt, {"run_id": 3, "slot": index})
                self.clock.advance(.05)
            writer.flush()
            writer.write_execution_end(execution_snapshot(2))
            writer.write_termination("operator", [], None, expected_frames=2)
        finally:
            writer.close()
        if invalid:
            with h5py.File(staging / "data.h5", "r+") as data:
                if invalid == "missing":
                    del data["hand_current"]
                elif invalid == "nan":
                    data["hand_current"][1, 0] = np.nan
                elif invalid == "geometry":
                    del data["meta"].attrs["camera_T_xarm_base_from_color"]
                elif invalid == "dispatch":
                    data["dispatch_status"][1] = (0, 0)
                elif invalid == "depth_rows":
                    data["depth"].resize(1, axis=0)
                elif invalid == "sdk":
                    del data["arm_execution/end"]
                    data["arm_execution"].create_dataset("end", data=json.dumps(dict(execution_snapshot(2), first_accepted=1)))
                elif invalid == "empty_cloud":
                    data["depth"][:] = 0
                elif invalid == "timing":
                    data["timestamp"][1] = .2
        destination = staging.with_name(name)
        staging.rename(destination)
        return destination

    def exporter(self):
        cloud = PointCloudConfig(remove_table=False, num_points=4,
                                 workspace=(-1., -1., 0., 1., 1., 1.),
                                 depth_support_min_neighbors=0, edge_support_min_neighbors=0,
                                 outlier_min_neighbors=0, outlier_min_component_points=1)
        return CanonicalExporter(ProcessingConfig(pointcloud=cloud), chunk_frames=1)

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "full numerical export requires optional Pinocchio")
    def test_full_export_preserves_15_fields_geometry_tactile_and_raw(self):
        import pinocchio

        source = self.fixture("episode_valid")
        before = file_contents(source)
        target = self.root / "canonical.zarr"
        exporter = self.exporter()
        result = exporter.export(source, target)
        self.assertEqual(result["total_frames"], 2)
        store = zarr.open_group(str(target), mode="r")
        specs = canonical_array_specs(4, 4, 4)
        self.assertEqual(len(specs), 15)
        self.assertEqual(set(store["data"].array_keys()), set(specs))
        for name, (tail, dtype) in specs.items():
            self.assertEqual(store["data"][name].shape, (2, *tail))
            self.assertEqual(store["data"][name].dtype, dtype)
            self.assertTrue(np.isfinite(store["data"][name][:]).all())
        arm_model = pinocchio.buildModelFromUrdf(str(XARM7_XHAND_COLLISION_URDF_PATH))
        hand_model = pinocchio.buildModelFromUrdf(str(XHAND_RIGHT_URDF_PATH))

        def placements(model, qpos, names):
            data = model.createData()
            pinocchio.forwardKinematics(model, data, qpos)
            pinocchio.updateFramePlacements(model, data)
            return [data.oMf[model.getFrameId(name)].copy() for name in names]

        def pose_values(placement):
            return np.concatenate((placement.translation, placement.rotation[:, 0],
                                   placement.rotation[:, 1]))

        config = exporter.processing
        mount_quat = config.handbase_quat_eef_wxyz
        mount = pinocchio.SE3(Rotation.from_quat((*mount_quat[1:], mount_quat[0])).as_matrix(),
                             np.asarray(config.handbase_position_eef_m))
        with EpisodeReader(source) as reader:
            np.testing.assert_array_equal(reader["timestamp"][:], [0., .05])
            np.testing.assert_array_equal(np.diff(reader.read_timing_rows(("observation_ns",))["observation_ns"]),
                                          [50_000_000])
            self.assertEqual(store.attrs["dt"], reader.dt)
            self.assertEqual(store.attrs["dt"], .05)
            intrinsic = reader.meta["camera_color_intrinsics"].reshape(3, 3)
            transform = reader.meta["camera_T_xarm_base_from_color"].reshape(4, 4)
            for index, rgb in enumerate(reader.iter_rgb_frames()):
                arm, hand = reader["arm_qpos"][index], reader["hand_qpos"][index]
                arm_goal = reader["action_arm_joint_target"][index]
                hand_goal = reader["action_hand_joint_target"][index]
                np.testing.assert_array_equal(store["data/joint_state"][index],
                                              np.concatenate((arm, hand)).astype(np.float32))
                np.testing.assert_array_equal(store["data/action"][index],
                                              np.concatenate((arm_goal, hand_goal)).astype(np.float32))
                for canonical, raw in (("arm_qvel", "arm_qvel"), ("arm_effort", "arm_effort"),
                                       ("hand_current", "hand_current"),
                                       ("finger_tactile_force", "hand_contact"),
                                       ("taxel_tactile_force", "hand_tactile_force")):
                    np.testing.assert_array_equal(store["data/" + canonical][index],
                                                  reader[raw][index].astype(np.float32))
                # Direct URDF FK distinguishes measured state from the submitted target.
                measured, = placements(arm_model, arm, ("custom_eef_link",))
                commanded, = placements(arm_model, arm_goal, ("custom_eef_link",))
                np.testing.assert_allclose(store["data/eef_pose"][index], pose_values(measured), atol=1e-7)
                np.testing.assert_allclose(store["data/action_ee"][index],
                                          np.concatenate((pose_values(commanded), hand_goal)), atol=1e-7)
                hand_urdf = np.empty(hand_model.nq)
                for joint, name in zip(hand_model.joints[1:], hand_model.names[1:]):
                    hand_urdf[joint.idx_q] = hand[XHAND_SDK_JOINT_NAMES.index(name)]
                tips = placements(hand_model, hand_urdf, config.fingertip_link_names)
                expected_tips = [(measured * mount).act(tip.translation) for tip in tips]
                np.testing.assert_allclose(store["data/fingertip_xyz"][index], expected_tips, atol=1e-7)
                np.testing.assert_array_equal(store["data/camera_intrinsic"][index],
                                              intrinsic.astype(np.float32).ravel())
                np.testing.assert_allclose(store["data/camera_extrinsic"][index],
                                          np.linalg.inv(transform)[:3].ravel(), atol=1e-7)
                depth = reader["depth"][index]
                np.testing.assert_array_equal(store["data/rgb"][index], rgb)
                np.testing.assert_array_equal(store["data/depth"][index], depth)
                # Each sampled point and its color must come from the same decoded RGB-D pixel.
                v, u = np.indices(depth.shape)
                pixels = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
                points = (pixels @ np.linalg.inv(intrinsic).T
                          * depth.reshape(-1, 1) * reader.meta["depth_scale"])
                points = points @ transform[:3, :3].T + transform[:3, 3]
                cloud = store["data/point_cloud"][index]
                distances = np.linalg.norm(cloud[:, None, :3] - points[None], axis=-1)
                matches = distances.argmin(axis=1)
                np.testing.assert_allclose(cloud[:, :3], points[matches], atol=1e-6)
                np.testing.assert_allclose(cloud[:, 3:], rgb.reshape(-1, 3)[matches] / 255., atol=1e-6)
        recipe = json.loads(json.dumps(asdict(config.pointcloud)))
        recipe["sampling_method"] = SAMPLING_METHOD
        self.assertEqual(store.attrs["pointcloud_config"], recipe)
        np.testing.assert_array_equal(store["meta/episode_ends"][:], [2])
        self.assertEqual(file_contents(source), before)
        canonical_before = file_contents(target)
        with self.assertRaises(FileExistsError):
            self.exporter().export(source, target)
        self.assertEqual(file_contents(source), before)
        self.assertEqual(file_contents(target), canonical_before)

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "full numerical export requires optional Pinocchio")
    def test_bad_episodes_are_rejected_whole_without_raw_changes(self):
        self.fixture("episode_valid")
        issues = ("missing", "nan", "geometry", "dispatch", "depth_rows", "sdk", "empty_cloud", "timing")
        for issue in issues:
            self.fixture("episode_" + issue, invalid=issue)
        before = file_contents(self.root / "raw")
        result = self.exporter().export(self.root / "raw", self.root / "canonical.zarr")
        self.assertEqual(result["episode_count"], 1)
        self.assertEqual(result["total_frames"], 2)
        self.assertEqual(set(result["rejected_episodes"]), {"episode_" + issue for issue in issues})
        self.assertTrue(all(result["rejected_episodes"].values()))
        self.assertEqual(file_contents(self.root / "raw"), before)

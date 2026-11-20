"""覆盖 HOI 点云几何、逐帧适配器及完整训练数据导出。"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.perception import HOI_NUM_POINTS, HOITrackingConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import (
    ARM_ACTION_SEMANTICS,
    CAMERA_TIMING_DTYPE,
    CAMERA_TIMING_FIELDS,
    TIMING_ROW_DTYPE,
    RawDataError,
    RawReader,
    camera_metadata,
)
from dexmani_real.deployment.rollout import EpisodeResult, Rollout
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.hoi import build_tracking_config, track_hoi_frame
from dexmani_real.perception.hoi_worker import run_hoi_worker
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES

BACKEND = Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
with patch.object(sys, "path", [str(BACKEND), *sys.path]):
    from cordvip_geometry.tracking import SceneResult, TrackingConfig, TrackingFailure

    from scripts.cordvip import annotate_episodes as annotation


def object_template():
    points = np.empty((HOI_NUM_POINTS, 3), dtype=np.float32)
    points[:256] = [1, 2, 3]
    points[256:] = [-1, 4, 2]
    return {
        "points_local": points,
        "object_names": np.array(["cup", "tray"]),
        "offsets": np.array([0, 256, HOI_NUM_POINTS]),
    }


def camera_snapshot():
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.5, -0.25, 1]
    intrinsics = CameraIntrinsics(2, 2, 2.0, 2.0, 0.5, 0.5, "none", (0.0,) * 5)
    return CameraSnapshot(None, intrinsics, 0.001, extrinsic)


def scene_result(*, tray_valid=True, motion=0):
    cup = np.eye(4)
    tray = np.eye(4)
    cup[:3, 3] = [0.125, motion, 0.25]
    tray[:3, 3] = [0.75, 0.25, motion]
    return SceneResult(
        poses={"tray": tray, "cup": cup} if tray_valid else {"cup": cup},
        valid={"tray": tray_valid, "cup": True},
        quality={"tray": {"depth_pixels": 20}, "cup": {"depth_pixels": 40}},
        failures={} if tray_valid else {"tray": "occluded"},
    )


class ObjectGeometryTest(unittest.TestCase):
    def test_identity_and_translation(self):
        template = object_template()
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        cloud = transform_task_points(template, poses, np.eye(4))
        np.testing.assert_array_equal(cloud, template["points_local"])
        poses[:, :3, 3] = [0.5, -1, 2]
        base = np.eye(4)
        base[:3, 3] = [2, 4, 1]
        cloud = transform_task_points(template, poses, base)
        np.testing.assert_array_equal(cloud, template["points_local"] + [2.5, 3, 3])
        self.assertEqual(cloud.dtype, np.float32)

    def test_rotations_compose_in_camera_order_and_keep_segments(self):
        base = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]])
        cup = np.array([[1, 0, 0, 2], [0, 0, -1, -1], [0, 1, 0, 3], [0, 0, 0, 1]])
        tray = np.eye(4)
        tray[:3, 3] = [-2, 3, 1]
        cloud = transform_task_points(object_template(), np.stack((cup, tray)), base)
        self.assertEqual(cloud.shape, (HOI_NUM_POINTS, 3))
        self.assertTrue(np.isfinite(cloud).all())
        np.testing.assert_array_equal(cloud[:256], np.tile([8, 8, 11], (256, 1)))
        np.testing.assert_array_equal(cloud[256:], np.tile([-3, 2, 9], (768, 1)))


class TrackFrameTest(unittest.TestCase):
    def test_initialize_and_track_normalize_in_template_order(self):
        template = object_template()
        tracker = Mock()
        frame, prompts = object(), {"cup": "a cup", "tray": "a tray"}
        initial = tracker.initialize.return_value = scene_result()
        tracker.track.return_value = scene_result(tray_valid=False)

        result = track_hoi_frame(tracker, frame, template, prompts, initialize=True)
        tracker.initialize.assert_called_once_with(frame, prompts)
        self.assertTrue(result.valid)
        self.assertEqual(result.object_pose_camera.dtype, np.float32)
        self.assertEqual(result.object_valid.dtype, np.dtype(bool))
        np.testing.assert_array_equal(
            result.object_pose_camera, [initial.poses["cup"], initial.poses["tray"]]
        )
        self.assertEqual(result.quality, ({"depth_pixels": 40}, {"depth_pixels": 20}))
        self.assertEqual(result.failures, {})

        result = track_hoi_frame(tracker, frame, template, prompts)
        tracker.track.assert_called_once_with(frame)
        self.assertFalse(result.valid)
        np.testing.assert_array_equal(result.object_valid, [True, False])
        np.testing.assert_array_equal(result.object_pose_camera[0], initial.poses["cup"])
        self.assertTrue(np.isnan(result.object_pose_camera[1]).all())
        self.assertEqual(result.quality, ({"depth_pixels": 40}, {"depth_pixels": 20}))
        self.assertEqual(result.failures, {"tray": "occluded"})

        tracker.track.return_value.poses["tray"] = initial.poses["tray"]
        stale_pose_result = track_hoi_frame(tracker, frame, template, prompts)
        self.assertFalse(stale_pose_result.valid)
        self.assertTrue(np.isnan(stale_pose_result.object_pose_camera[1]).all())

    def test_initialization_failure_returns_invalid_frame(self):
        template = object_template()
        tracker = Mock()
        tracker.initialize.side_effect = TrackingFailure("registration failed")
        result = track_hoi_frame(tracker, object(), template, {}, initialize=True)
        self.assertFalse(result.valid)
        np.testing.assert_array_equal(result.object_valid, [False, False])
        self.assertTrue(np.isnan(result.object_pose_camera).all())
        self.assertEqual(result.quality, ({}, {}))
        self.assertEqual(result.failures, {"cup": "registration failed", "tray": "registration failed"})

        tracker.initialize.side_effect = RuntimeError("backend failed")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            track_hoi_frame(tracker, object(), template, {}, initialize=True)


class TrackingConfigTest(unittest.TestCase):
    def test_backend_default_and_overrides(self):
        self.assertEqual(build_tracking_config(RealConfig().hoi_tracking), TrackingConfig())
        tracking = HOITrackingConfig(segmentation_interval_frames=3, recovery_enabled=False)
        self.assertEqual(
            build_tracking_config(tracking),
            TrackingConfig(segmentation_interval_frames=3, recovery_enabled=False),
        )

    def test_ostrack_defaults_and_explicit_overrides(self):
        config = build_tracking_config(HOITrackingConfig())
        self.assertEqual(config.recovery_retry_interval_frames, 8)
        self.assertEqual(config.ostrack_checkpoint, "third_party/weights/ostrack/OSTrack_ep0300.pth.tar")
        config = build_tracking_config(HOITrackingConfig(
            ostrack_checkpoint="/tmp/checkpoint.pth.tar",
            lost_timeout_s=0.4, recovery_retry_interval_frames=5, xy_prior_gain=0,
        ))
        self.assertEqual(config.ostrack_checkpoint, "/tmp/checkpoint.pth.tar")
        self.assertEqual(config.lost_timeout_s, 0.4)
        self.assertEqual(config.recovery_retry_interval_frames, 5)
        self.assertEqual(config.xy_prior_gain, 0)

    def test_effective_tracking_config_rejects_invalid_values(self):
        for field, value in (
            ("segmentation_interval_frames", 0),
            ("recovery_enabled", 1),
            ("recovery_retry_interval_frames", 0),
            ("xy_prior_gain", -0.1),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                build_tracking_config(HOITrackingConfig(**{field: value}))

    def test_backend_config_cannot_bypass_experiment_settings(self):
        backend = TrackingConfig(min_depth_pixels=20)
        with self.assertRaisesRegex(TypeError, "HOITrackingConfig"):
            build_tracking_config(backend)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "HOITrackingConfig"):
            RealConfig(hoi_tracking=backend)  # type: ignore[arg-type]

    def test_annotation_cli_loads_shared_experiment(self):
        with TemporaryDirectory() as directory:
            experiment = Path(directory) / "experiment.py"
            experiment.write_text(
                "from dexmani_real.config.perception import HOITrackingConfig\n"
                "from dexmani_real.config.real import RealConfig\n"
                "CONFIG = RealConfig(hoi_tracking=HOITrackingConfig(7, False))\n",
                encoding="utf-8",
            )
            episode = Path(directory) / "episode_fixture"
            episode.mkdir()
            (episode / "data.h5").touch()
            args = [str(episode), "--task-dir", directory, "--config", str(experiment)]
            with patch.object(annotation, "annotate_episodes") as annotate:
                self.assertEqual(annotation.main(args), 0)
            self.assertEqual(annotate.call_args.kwargs["tracking_config"], HOITrackingConfig(7, False))


class HOIAdapterTest(unittest.TestCase):
    def run_worker(self, tracker, camera, template, prompts, camera_rows, request, config):
        running = SimpleNamespace(value=True)
        frames = iter(camera_rows)
        request_lock = Lock()
        published = []

        def read_frame():
            result = next(frames, None)
            if result is None:
                running.value = False
            return result

        def write_result(**fields):
            self.assertTrue(request_lock.locked())
            published.append(fields)

        shared = SimpleNamespace(
            camera_ring=SimpleNamespace(read=read_frame),
            hoi_ring=SimpleNamespace(field_names={"rgb"}, write=write_result),
            hoi_request=request,
            hoi_request_lock=request_lock,
            is_running=running,
            hoi_ready=SimpleNamespace(value=False),
        )
        effective_config = build_tracking_config(config)
        with (
            patch("cordvip_geometry.SceneTracker", return_value=tracker) as factory,
            patch("dexmani_real.perception.hoi_worker.signal.signal"),
            patch("dexmani_real.perception.hoi_worker.time.sleep"),
            patch(
                "dexmani_real.perception.hoi_worker.transform_task_points", wraps=transform_task_points
            ) as transform,
        ):
            run_hoi_worker(shared, camera, template, {}, prompts, effective_config)
        self.assertTrue(shared.hoi_ready.value)
        self.assertIs(factory.call_args.kwargs["config"], effective_config)
        return published, [call.args[1] for call in transform.call_args_list]

    def test_worker_and_raw_annotation_agree_after_retry_and_recovery(self):
        camera, template = camera_snapshot(), object_template()
        prompts = {"cup": "a cup", "tray": "a tray"}
        config = HOITrackingConfig(segmentation_interval_frames=4)
        sequences, stamps = [12, 12, 13, 14, 15], [120, 120, 130, 140, 150]
        timing = np.zeros(len(sequences), dtype=TIMING_ROW_DTYPE)
        timing["camera_sequence"], timing["camera_ns"] = sequences, stamps
        rgb = [np.full((2, 2, 3), sequence, dtype=np.uint8) for sequence in sequences]
        depth = np.full((len(sequences), 2, 2), 400, dtype=np.uint16)
        camera_rows = [
            ({"rgb": image, "depth": pixels, "timestamp_ns": stamp}, sequence)
            for image, pixels, stamp, sequence in zip(rgb, depth, stamps, sequences, strict=True)
        ]
        current = {"episode_id": 7, "after_ns": 110, "after_sequence": 10}
        request = SimpleNamespace(latest_sequence=3, read=Mock(return_value=(current, 3)))

        tracker = Mock()
        initial = scene_result()
        partial = scene_result(tray_valid=False)
        recovered = scene_result(motion=0.5)
        tracker.initialize.side_effect = [TrackingFailure("missing detection"), initial]
        tracker.track.side_effect = [partial, recovered]
        discarded = [(camera_rows[0][0], 10), ({**camera_rows[0][0], "timestamp_ns": 110}, 11)]
        with self.assertLogs("dexmani_real.perception.hoi_worker", level="WARNING"):
            online, online_poses = self.run_worker(
                tracker, camera, template, prompts, [*discarded, *camera_rows], request, config
            )
        online_calls = list(tracker.method_calls)

        tracker.reset_mock()
        tracker.initialize.side_effect = [TrackingFailure("missing detection"), initial]
        tracker.track.side_effect = [partial, recovered]
        reader = MagicMock(
            spec=RawReader,
            num_frames=len(sequences),
            read_camera=lambda: camera,
            read_timing=Mock(return_value=timing),
            iter_rgb_frames=lambda: iter(rgb),
        )
        reader.__getitem__.side_effect = {"depth": depth}.__getitem__
        with redirect_stdout(io.StringIO()):
            poses, extrinsic, camera_timing, quality, valid = annotation.track_episode(
                reader, tracker, template, prompts
            )

        self.assertEqual(
            [call[0] for call in online_calls], ["initialize", "initialize", "track", "track"]
        )
        self.assertEqual([call[0] for call in tracker.method_calls], [call[0] for call in online_calls])
        for online_call, offline_call in zip(online_calls, tracker.method_calls, strict=True):
            live_frame, raw_frame = online_call.args[0], offline_call.args[0]
            self.assertEqual(live_frame.source_sequence, raw_frame.source_sequence)
            self.assertEqual(live_frame.timestamp_ns, raw_frame.timestamp_ns)
            np.testing.assert_array_equal(live_frame.rgb, raw_frame.rgb)
            np.testing.assert_array_equal(live_frame.depth_m, raw_frame.depth_m)

        np.testing.assert_array_equal(camera_timing["camera_sequence"], sequences)
        np.testing.assert_array_equal(camera_timing["camera_ns"], stamps)
        np.testing.assert_array_equal(
            valid, [[False, False], [False, False], [True, True], [True, False], [True, True]]
        )
        np.testing.assert_array_equal(online_poses, poses[valid.all(axis=1)])
        self.assertTrue(np.isnan(poses[:2]).all())
        self.assertTrue(np.isnan(poses[3, 1]).all())
        self.assertEqual(quality[0], ({}, {}))
        self.assertEqual(quality[1], quality[0])
        self.assertEqual(quality[3], ({"depth_pixels": 40}, {"depth_pixels": 20}))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "hoi.h5"
            tracking = asdict(build_tracking_config(config))
            published_path = write_hoi(
                str(path),
                template,
                poses,
                extrinsic,
                camera_timing=camera_timing,
                quality=quality,
                tracking_config=tracking,
                object_valid=valid,
            )
            self.assertIsInstance(published_path, Path)
            self.assertEqual(published_path, path)
            self.assertTrue(published_path.is_file())
            with h5py.File(published_path, "r") as saved:
                self.assertEqual(json.loads(saved.attrs["tracking_config"]), tracking)
                np.testing.assert_array_equal(saved["object_pose_camera"][:], poses)
                np.testing.assert_array_equal(saved["object_valid"][:], valid)
                np.testing.assert_array_equal(saved["camera_timing"][:], camera_timing)
                np.testing.assert_array_equal(saved["quality/depth_pixels"][2:], [[40, 20]] * 3)
                self.assertTrue(np.isnan(saved["quality/depth_pixels"][:2]).all())
                clouds = saved["object_pointcloud"][:]

        self.assertEqual(len(online), 4)
        for published, raw_index in zip(online, [0, 2, 3, 4], strict=True):
            self.assertEqual(int(published["episode_id"]), 7)
            self.assertEqual(published["timestamp_ns"], stamps[raw_index])
            self.assertEqual(published["source_camera_sequence"], sequences[raw_index])
            self.assertEqual(published["valid"], bool(valid[raw_index].all()))
            np.testing.assert_array_equal(published["rgb"], rgb[raw_index])
            np.testing.assert_array_equal(published["object_pointcloud"], clouds[raw_index])
        np.testing.assert_array_equal(clouds[[0, 1, 3]], 0)
        self.assertTrue(np.isfinite(clouds[[2, 4]]).all())
        np.testing.assert_array_equal(poses[2], [initial.poses["cup"], initial.poses["tray"]])
        np.testing.assert_array_equal(poses[4], [recovered.poses["cup"], recovered.poses["tray"]])

    def test_worker_drops_inflight_result_and_reinitializes_new_request(self):
        command = {"episode_id": 2, "after_ns": 100, "after_sequence": 8}
        request = SimpleNamespace(latest_sequence=1, read=Mock(return_value=(command, 1)))
        tracker = Mock()

        def initialize(frame, prompts):
            if frame.source_sequence == 9:
                request.latest_sequence = 2
                replacement = {"episode_id": 3, "after_ns": 110, "after_sequence": 9}
                request.read.return_value = (replacement, 2)
            return scene_result()

        tracker.initialize.side_effect = initialize
        row = {
            "rgb": np.zeros((2, 2, 3), np.uint8),
            "depth": np.zeros((2, 2), np.uint16),
            "timestamp_ns": 110,
        }
        published, _ = self.run_worker(
            tracker, camera_snapshot(), object_template(), {},
            [(row, 9), (dict(row, timestamp_ns=120), 10)], request, HOITrackingConfig(),
        )
        self.assertEqual(tracker.initialize.call_count, 2)
        tracker.track.assert_not_called()
        self.assertEqual([packet["episode_id"] for packet in published], [3])

    def test_rollout_records_error_and_stops_if_request_lock_is_held(self):
        rollout = Rollout.__new__(Rollout)
        rollout.result, rollout.started_ns = EpisodeResult(index=1), 0
        rollout.preparation_issue = None
        rollout.runtime_state = SimpleNamespace(stop_signal=Event())
        rollout.observations = SimpleNamespace(needs=SimpleNamespace(hoi=True))
        request = Mock()

        def locked(*, timeout):
            self.assertTrue(rollout.runtime_state.stop_signal.is_set())
            self.assertGreater(timeout, 0)
            return False

        lock = Mock(acquire=Mock(side_effect=locked))
        rollout.sensors = SimpleNamespace(channels=SimpleNamespace(hoi_request_lock=lock, hoi_request=request))
        with (
            patch.object(rollout, "_clear_policy_context"),
            self.assertRaisesRegex(TimeoutError, "HOI request lock timed out"),
        ):
            rollout.fail("control", RuntimeError("HOI worker exited"))
        self.assertEqual(rollout.result.errors[0]["message"], "HOI worker exited")
        request.write.assert_not_called()
        lock.release.assert_not_called()


class HOIExportTest(unittest.TestCase):
    def test_camera_timing_requires_consistent_sequence_timestamps(self):
        rows = np.zeros(3, dtype=TIMING_ROW_DTYPE)
        rows["control_ns"] = rows["observation_ns"] = [100, 110, 120]
        rows["camera_sequence"] = [7, 7, 8]
        rows["camera_ns"] = [80, 80, 90]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as raw:
                raw.create_group("meta").attrs["termination_reason"] = "completed"
                raw.create_dataset("timing/rows", data=rows)
            with RawReader(directory) as reader:
                np.testing.assert_array_equal(reader.read_timing(source_clocks=("camera_ns",)), rows)

            rows["camera_ns"][1] = 81
            with h5py.File(path, "r+") as raw:
                raw["timing/rows"][1] = rows[1]
            with RawReader(directory) as reader, self.assertRaisesRegex(RawDataError, "camera sequence or timestamp at row 1"):
                reader.read_timing(source_clocks=("camera_ns",))

    def test_annotation_publish_failure_preserves_existing_file_and_cleans_temporary_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hoi.h5"
            original = b"existing annotation"
            path.write_bytes(original)

            with self.assertRaises(FileExistsError):
                write_hoi(
                    path,
                    object_template(),
                    np.broadcast_to(np.eye(4), (1, 2, 4, 4)),
                    np.eye(4),
                    camera_timing=np.array([(100, 1)], dtype=CAMERA_TIMING_DTYPE),
                    quality=[({}, {})],
                    tracking_config={},
                    object_valid=np.ones((1, 2), dtype=bool),
                )

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_annotation_reuse_rejects_missing_datasets(self):
        camera = camera_snapshot()
        timing = np.zeros(1, dtype=TIMING_ROW_DTYPE)
        camera_timing = timing[list(CAMERA_TIMING_FIELDS)].astype(CAMERA_TIMING_DTYPE)
        with TemporaryDirectory() as directory:
            episode = Path(directory)
            with h5py.File(episode / "data.h5", "w") as raw:
                raw.create_group("meta").attrs.update(
                    termination_reason="completed", **camera_metadata(camera),
                )
                raw.create_dataset("timing/rows", data=timing)
            path = episode / "hoi.h5"
            for field in ("base_from_camera", "object_pose_camera", "object_pointcloud", "quality"):
                with self.subTest(field=field):
                    write_hoi(
                        path, object_template(), np.broadcast_to(np.eye(4), (1, 2, 4, 4)),
                        camera.T_xarm_base_from_color, camera_timing=camera_timing, quality=[({}, {})],
                        tracking_config={}, object_valid=np.ones((1, 2), dtype=bool), overwrite=True,
                    )
                    with h5py.File(path, "r+") as annotated:
                        del annotated[field]
                    with self.assertRaisesRegex(RawDataError, f"HOI {field}"):
                        annotation.annotate_episodes([episode], episode, tracking_config=HOITrackingConfig())

            with h5py.File(path, "r+") as annotated:
                annotated.create_group("quality").create_dataset("depth_pixels", shape=(1, 1), dtype=float)
            with RawReader(episode) as reader, self.assertRaisesRegex(RawDataError, "HOI quality/depth_pixels"):
                reader.read_hoi_validity()

    def test_complete_hoi_export_uses_raw_row_joint_state_and_fk(self):
        template, camera = object_template(), camera_snapshot()
        frames = 5
        ms = 1_000_000
        timings = np.zeros(frames, dtype=TIMING_ROW_DTYPE)
        timings["arm_ns"] = np.array([1000, 1020, 1040, 1060, 1080]) * ms
        timings["hand_ns"] = np.array([1000, 1024, 1040, 1064, 1084]) * ms
        # 第二段的末帧相机时间晚于所有关节反馈。
        timings["camera_ns"] = np.array([1000, 1030, 1040, 1070, 1090]) * ms
        timings["camera_sequence"] = np.arange(1, frames + 1)
        timings["control_ns"] = np.array([1006, 1034, 1046, 1076, 1094]) * ms
        timings["observation_ns"] = timings["control_ns"] + ms

        arm_joints = np.arange(frames)[:, None] * np.full((1, len(XARM7_JOINT_NAMES)), 0.02)
        hand_joints = np.arange(frames)[:, None] * np.full((1, len(XHAND_SDK_JOINT_NAMES)), 0.04)
        arm_target = np.full_like(arm_joints, np.nan)
        hand_target = np.full_like(hand_joints, np.nan)
        arm_target[1], hand_target[1] = arm_joints[1] + 0.01, hand_joints[1] + 0.01
        camera_timing = timings[list(CAMERA_TIMING_FIELDS)].astype(CAMERA_TIMING_DTYPE)
        scenes = [scene_result(motion=0.25 * row) for row in range(frames)]
        poses = np.array([[scene.poses[name] for name in template["object_names"]] for scene in scenes])
        with TemporaryDirectory() as directory:
            episode = Path(directory) / "episode_fixture"
            episode.mkdir()
            with h5py.File(episode / "data.h5", "w") as raw:
                raw.create_group("meta").attrs.update(
                    termination_reason="completed",
                    task_label="fixture",
                    control_hz=50.0,
                    joint_units="rad",
                    arm_action_semantics=ARM_ACTION_SEMANTICS,
                    arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
                    hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
                    **camera_metadata(camera),
                )
                raw.create_dataset("timing/rows", data=timings)
                raw.create_dataset("timing/segment_starts", data=np.array([0, 3], dtype=np.int64))
                raw.create_dataset("arm_qpos", data=arm_joints)
                raw.create_dataset("hand_qpos", data=hand_joints)
                raw.create_dataset("action_arm_joint_target", data=arm_target)
                raw.create_dataset("action_hand_joint_target", data=hand_target)
            write_hoi(
                episode / "hoi.h5",
                template,
                poses,
                camera.T_xarm_base_from_color,
                camera_timing=camera_timing,
                quality=[({}, {})] * frames,
                tracking_config=asdict(TrackingConfig()),
                object_valid=np.ones((frames, 2), dtype=bool),
            )
            with RawReader(episode) as reader:
                np.testing.assert_array_equal(
                    reader.read_timing(source_clocks=("camera_ns", "arm_ns", "hand_ns")), timings
                )
                cloud = reader["object_pointcloud"][:]
            output = Path(directory) / "fixture.zarr"
            fields = (
                "joint_state",
                "eef_pose",
                "fingertip_xyz",
                "hand_pointcloud",
                "object_pointcloud",
                "action",
                "action_ee",
            )
            with patch("dexmani_real.data.export.MAX_BATCH_FRAMES", 2):
                result = export_zarr(episode, output, processing=ExportConfig(), fields=fields)
            self.assertEqual(result["total_frames"], frames)
            root = zarr.open_group(str(output), mode="r")
            data = root["data"]
            self.assertEqual(set(data), set(fields))
            for name in fields:
                self.assertEqual(data[name].dtype, np.float32)
            np.testing.assert_array_equal(root["meta/episode_ends"][:], [3, 5])
            self.assertEqual(
                root.attrs["data_processing"]["object_pointcloud"],
                {"joint_state_source": "raw_observation_row"},
            )
            np.testing.assert_array_equal(
                data["joint_state"][:], np.column_stack((arm_joints, hand_joints)).astype(np.float32)
            )
            np.testing.assert_array_equal(data["object_pointcloud"][:], cloud)

            arm_fk, hand_fk = ArmFK(), HandFK()
            eef = eef_poses(arm_joints, arm_fk=arm_fk)
            expected_geometry = hand_geometry(
                eef,
                hand_joints,
                arm_fk=arm_fk,
                hand_fk=hand_fk,
                hand_cloud=HandPointCloud(hand_fk, HOI_NUM_POINTS),
            )
            np.testing.assert_allclose(data["eef_pose"][:], encode_eef_pose(eef), atol=1e-7)
            for name, expected in expected_geometry.items():
                np.testing.assert_allclose(data[name][:], expected, atol=1e-7)

            # 等待行沿用本段已提交的目标，每段开头使用该段首行的原始关节反馈。
            expected_arm_target = np.stack(
                (arm_joints[0], arm_target[1], arm_target[1], arm_joints[3], arm_joints[3])
            )
            expected_hand_target = np.stack(
                (hand_joints[0], hand_target[1], hand_target[1], hand_joints[3], hand_joints[3])
            )
            expected_action = np.column_stack((expected_arm_target, expected_hand_target)).astype(np.float32)
            np.testing.assert_array_equal(data["action"][:], expected_action)
            target_poses = eef_poses(expected_arm_target, arm_fk=arm_fk)
            np.testing.assert_allclose(
                data["action_ee"][:],
                np.column_stack((encode_eef_pose(target_poses), expected_hand_target)),
                atol=1e-7,
            )
            object_action_output = Path(directory) / "object_action.zarr"
            export_zarr(
                episode, object_action_output, processing=ExportConfig(), fields=("object_pointcloud", "action"),
            )
            object_action = zarr.open_group(str(object_action_output), mode="r")
            self.assertNotIn("object_pointcloud", object_action.attrs["data_processing"])
            np.testing.assert_array_equal(object_action["data/action"][:], expected_action)


if __name__ == "__main__":
    unittest.main()

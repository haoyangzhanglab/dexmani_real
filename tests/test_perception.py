import json
import shutil
import threading
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from dexmani_real.config import load_experiment_config
from dexmani_real.data.canonical.export import export_canonical
from dexmani_real.data.canonical.processing import iter_canonical_blocks
from dexmani_real.data.canonical.schema import ExportConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.data.raw.schema import (
    DATASET_SPECS,
    TACTILE_AXES,
    TACTILE_FINGER_ORDER,
    TACTILE_PROCESSING,
    TACTILE_UNITS,
)
from dexmani_real.data.raw.writer import EpisodeWriter
from dexmani_real.deployment.observation import (
    PolicyObservationRow,
    build_observation_kinematics,
    build_policy_observation,
    policy_dependencies,
)
from dexmani_real.deployment.config import _restore_cloud_config
from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    CameraSnapshot,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.geometry.pose import quat_wxyz_to_rotmat
from dexmani_real.perception import worker
from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.perception.hand_cloud import HandPointCloud
from dexmani_real.perception.pointcloud import (
    PointCloudProcessor,
    filter_depth,
    radius_inliers,
    sample_indices,
    voxel_downsample,
)
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE


class PointCloudTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraIntrinsics(40, 40, 100.0, 100.0, 19.5, 19.5, "none", (0, 0, 0, 0, 0))
        self.config = PointCloudConfig(
            num_points=256,
            workspace=(-1, -1, 0, 1, 1, 1),
        )
        rows, columns = np.indices((40, 40))
        self.rgb = np.stack((rows * 5, columns * 5, np.full_like(rows, 200)), axis=-1).astype(
            np.uint8
        )
        x = (columns - self.camera.ppx) / self.camera.fx
        self.plane = (0.1, 0, 1, -0.5)
        height = np.where(columns < 20, 0, 0.02)
        depth_m = (0.5 + height * np.linalg.norm(self.plane[:3])) / (1 + 0.1 * x)
        self.depth = np.rint(depth_m * 1000).astype(np.uint16)

    def cloud(self, **changes):
        inputs = dict(
            depth_scale_m=0.001,
            intrinsics=self.camera,
            T_xarm_base_from_color=np.eye(4),
            table_plane_abcd=self.plane,
            config=self.config,
        )
        depth = changes.pop("depth_raw", self.depth)
        color = changes.pop("color", self.rgb)
        inputs.update(changes)
        return PointCloudProcessor(**inputs).process(depth, color)

    def test_tilted_table_crop_preserves_rgb_and_point_coordinates(self):
        cloud = self.cloud()
        self.assertEqual(cloud.shape, (256, 6))
        self.assertEqual(cloud.dtype, np.float32)
        rows = np.rint(cloud[:, 3] * 255 / 5).astype(int)
        columns = np.rint(cloud[:, 4] * 255 / 5).astype(int)
        self.assertTrue((columns >= 20).all())
        z = self.depth[rows, columns].astype(np.float32) * 0.001
        expected = np.column_stack(((columns - 19.5) * z / 100, (rows - 19.5) * z / 100, z))
        np.testing.assert_allclose(cloud[:, :3], expected, atol=1e-7)
        np.testing.assert_allclose(cloud[:, 3:], self.rgb[rows, columns] / 255, atol=1e-7)
        np.testing.assert_array_equal(
            cloud, self.cloud(table_plane_abcd=tuple(3 * np.asarray(self.plane)))
        )
        np.testing.assert_array_equal(cloud, self.cloud())

        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
        transform[:3, 3] = [0.2, -0.1, 0.1]
        plane_base = tuple(np.linalg.inv(transform).T @ self.plane)
        transformed = self.cloud(T_xarm_base_from_color=transform, table_plane_abcd=plane_base)
        self.assertTrue((transformed[:, 4] * 255 / 5 >= 20).all())
        height = (transformed[:, :3] @ plane_base[:3] + plane_base[3]) / np.linalg.norm(plane_base[:3])
        np.testing.assert_allclose(height, 0.02, atol=0.0006)

    def test_crop_uses_base_frame_and_depth_limits(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.3, -0.1, 0.2]
        config = replace(self.config, remove_table=False, workspace=(0.3, -1, 0, 1, 1, 1))
        cloud = self.cloud(config=config, T_xarm_base_from_color=transform, table_plane_abcd=None)
        self.assertTrue((cloud[:, 0] >= 0.3).all())
        self.assertTrue((cloud[:, 2] > 0.69).all())
        for depth in (
            np.zeros_like(self.depth),
            np.full_like(self.depth, 200),
            np.full_like(self.depth, 2000),
        ):
            self.assertIsNone(self.cloud(depth_raw=depth))
        sparse = np.zeros_like(self.depth)
        sparse[24, 24] = self.depth[24, 24]
        self.assertIsNone(self.cloud(depth_raw=sparse))
        self.assertIsNone(self.cloud(config=replace(self.config, workspace=(2, 2, 2, 3, 3, 3))))
        with self.assertRaisesRegex(ValueError, "calibrated table"):
            self.cloud(table_plane_abcd=None)
        with self.assertRaisesRegex(ValueError, "extrinsics"):
            self.cloud(T_xarm_base_from_color=None)

    def test_depth_edges_remove_intermediate_measurements(self):
        depth = np.array([[0.5, 0.5, 0.8], [0.5, 0.65, 0.8], [0.5, 0.8, 0.8]], dtype=np.float32)
        keep = filter_depth(depth)
        self.assertFalse(keep[1, 1])
        self.assertEqual(keep.sum(), 8)

    def test_processor_reuse_returns_fresh_clouds_and_no_stale_empty_frame(self):
        processor = PointCloudProcessor(
            depth_scale_m=0.001,
            intrinsics=self.camera,
            T_xarm_base_from_color=np.eye(4),
            table_plane_abcd=self.plane,
            config=self.config,
        )
        first = processor.process(self.depth, self.rgb)
        snapshot = first.copy()
        second = processor.process(self.depth + 50, 255 - self.rgb)
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.shares_memory(first, second))
        self.assertIsNone(processor.process(np.zeros_like(self.depth), self.rgb))
        np.testing.assert_array_equal(first, snapshot)
        np.testing.assert_array_equal(processor.process(self.depth, self.rgb), snapshot)

    def test_voxel_means_and_radius_neighbors(self):
        points = np.array([[0, 0, 0], [0.002, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        colors = np.eye(3, dtype=np.float32)
        xyz, rgb = voxel_downsample(points, colors)
        np.testing.assert_allclose(xyz, [[0.001, 0, 0], [0.1, 0, 0]])
        np.testing.assert_allclose(rgb, [[0.5, 0.5, 0], [0, 0, 1]])
        cluster = np.zeros((7, 3), dtype=np.float32)
        cluster[:, 0] = np.arange(7) * 0.001
        points = np.vstack((cluster, [0.1, 0, 0]))
        np.testing.assert_array_equal(radius_inliers(points), [True] * 7 + [False])
        self.assertFalse(radius_inliers(cluster[:6]).any())
        self.assertFalse(radius_inliers(points[-1:]).any())

    def test_fps_is_deterministic_and_padding_uses_observed_points(self):
        points = np.random.default_rng(1).uniform(size=(300, 3)).astype(np.float32)
        selected = sample_indices(points, 32)
        self.assertEqual(len(np.unique(selected)), 32)
        np.testing.assert_array_equal(selected, sample_indices(points, 32))
        np.testing.assert_array_equal(sample_indices(points[:3], 8), [0, 1, 2, 0, 1, 2, 0, 1])

    def test_config_roundtrip_requires_exact_current_fields(self):
        config = load_experiment_config(cli_overrides={"pointcloud.table_margin_m": 0.012})
        self.assertEqual(config.pointcloud.table_margin_m, 0.012)
        for config in (self.config, HandCloudConfig(num_points=17)):
            recipe = json.loads(json.dumps(asdict(config)))
            self.assertEqual(_restore_cloud_config(recipe, type(config)), config)
            extra = {"voxel_size_m": 0.01} if isinstance(config, PointCloudConfig) else {"seed": 0}
            with self.assertRaisesRegex(ValueError, "re-export"):
                _restore_cloud_config({**recipe, **extra}, type(config))
            del recipe["num_points"]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _restore_cloud_config(recipe, type(config))
        for change in (
            dict(num_points=0),
            dict(table_margin_m=-1),
            dict(table_margin_m=float("nan")),
            dict(workspace=(0, 0, 0, 0, 1, 1)),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)
        with self.assertRaises(TypeError):
            load_experiment_config(cli_overrides={"pointcloud.voxel_size_m": 0.01})

    def test_offline_mount_config_rejects_invalid_geometry(self):
        config = ExportConfig(pointcloud=self.config, table_plane_abcd=self.plane)
        for change in (
            dict(handbase_position_eef_m=(0.1,)),
            dict(handbase_position_eef_m=(np.nan, 0, 0)),
            dict(handbase_quat_eef_wxyz=(2, 0, 0, 0)),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config, **change)

    def test_raw_export_stops_on_bad_episode_and_supports_explicit_exclusion(self):
        config = ExportConfig(pointcloud=self.config, table_plane_abcd=self.plane)
        camera = CameraSnapshot("test", self.camera, 0.001, np.eye(4))
        arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "raw" / "episode_1"
            episode.mkdir(parents=True)
            writer = EpisodeWriter(episode, control_hz=16)
            try:
                writer.open({
                    **camera.metadata(),
                    "task_label": "geometry",
                    "collection_source": "test",
                    "hand_tactile_processing": TACTILE_PROCESSING,
                    "hand_tactile_units": TACTILE_UNITS,
                    "hand_tactile_axes": TACTILE_AXES,
                    "hand_tactile_finger_order": TACTILE_FINGER_ORDER,
                    "hand_tactile_hold_max_age_s": 0.1,
                })
                for index in range(2):
                    stamp = 1_000_000_000 + index * 62_500_000
                    arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
                    hand["tactile_aggregate_timestamp_ns"] = stamp
                    hand["tactile_dense_timestamp_ns"] = stamp
                    row = SimpleNamespace(
                        arm=arm, hand=hand, observation_timestamp_ns=stamp,
                        control_timestamp_ns=stamp,
                        camera=dict(rgb=self.rgb, depth=self.depth, timestamp_ns=stamp, ring_sequence=index),
                    )
                    command = SimpleNamespace(arm_qpos=arm["qpos"][0], hand_qpos=hand["qpos"][0])
                    writer.append_frame(row, command)
                writer.flush()
                writer.write_termination("operator", {}, expected_frames=2, technical_status="normal")
            finally:
                writer.close()

            rejected = shutil.copytree(episode, root / "raw" / "episode_0")
            with h5py.File(rejected / "data.h5", "r+") as data:
                data["depth"][1] = 0
            output = root / "canonical.zarr"
            with self.assertLogs("dexmani_real.data.canonical.export", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "point_cloud unavailable at row 1"):
                    export_canonical(root / "raw", root / "failed.zarr", processing=config)
            self.assertFalse((root / "failed.zarr").exists())
            self.assertEqual(len(list(root.glob(".failed.zarr.tmp-*"))), 1)
            with patch("dexmani_real.data.canonical.export.MAX_BATCH_FRAMES", 1):
                result = export_canonical(root / "raw", output, processing=config, exclude=("episode_0",))
            self.assertEqual(result["total_frames"], 2)
            self.assertEqual(result["excluded_episodes"], ["episode_0"])
            self.assertNotIn("rejected_episodes", result)
            store = zarr.open_group(str(output), mode="r")
            self.assertEqual(store.attrs["episode_ids"], ["episode_1"])
            np.testing.assert_array_equal(store["meta/episode_ends"][:], [2])
            for name, (shape, dtype) in canonical_array_specs(256, 40, 40, 1024).items():
                self.assertEqual(store[f"data/{name}"].shape, (2, *shape))
                self.assertEqual(store[f"data/{name}"].dtype, dtype)
            restored = _restore_cloud_config(store.attrs["pointcloud_config"], PointCloudConfig)
            self.assertEqual(restored, self.config)
            self.assertEqual(
                _restore_cloud_config(store.attrs["hand_pointcloud_config"], HandCloudConfig),
                config.hand_pointcloud,
            )
            with EpisodeReader(episode) as reader:
                decoded = next(reader.iter_rgb_frames())
            np.testing.assert_array_equal(store["data/point_cloud"][0], self.cloud(color=decoded, config=restored))
            self.assertFalse(list(root.glob(".canonical.zarr.tmp-*")))

    def test_worker_offline_and_policy_geometry_agree(self):
        arm, hand = np.array([0, -0.5, 0.1, 1, 0, 0.5, 0.1]), np.linspace(0.1, 0.5, 12)
        raw = {
            name: np.zeros((1, *shape), dtype=dtype)
            for name, (shape, dtype) in DATASET_SPECS.items()
        }
        raw.update(arm_qpos=arm[None], hand_qpos=hand[None], depth=self.depth[None])
        reader = MagicMock(num_frames=1, path=Path("/unused"))
        reader.__getitem__.side_effect = raw.__getitem__
        reader.iter_rgb_frames.return_value = iter([self.rgb])
        hand_config = HandCloudConfig()
        processing = ExportConfig(
            pointcloud=self.config, hand_pointcloud=hand_config, table_plane_abcd=self.plane
        )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("z", 15, degrees=True).as_matrix()
        transform[:3, 3] = (0.1, -0.2, 0.03)
        camera = CameraSnapshot("test", self.camera, 0.001, transform)
        hand_fk = HandFK()
        surface = HandPointCloud(hand_fk, hand_config.num_points)
        blocks = list(
            iter_canonical_blocks(
                reader,
                processing,
                arm_fk=ArmFK(),
                hand_fk=hand_fk,
                hand_cloud=surface,
                chunk_frames=1,
                camera=camera,
                handbase_rotation_eef=quat_wxyz_to_rotmat(processing.handbase_quat_eef_wxyz),
            )
        )
        block = blocks[0]
        specs = canonical_array_specs(256, 40, 40, 1024)
        self.assertEqual(set(block), set(specs))
        for name, (shape, dtype) in specs.items():
            self.assertEqual(block[name].shape, (1, *shape))
            self.assertEqual(block[name].dtype, dtype)
        np.testing.assert_allclose(
            block["camera_extrinsic"][0].reshape(3, 4),
            np.linalg.inv(transform)[:3],
            atol=1e-7,
        )

        channels = SimpleNamespace(
            is_running=SimpleNamespace(value=True),
            pointcloud_ready=threading.Event(),
            pointcloud_ring=SimpleNamespace(write=Mock()),
        )

        def read_frame(sequence):
            channels.is_running.value = False
            return {"depth": self.depth, "rgb": self.rgb, "timestamp_ns": 123}, sequence

        channels.camera_ring = SimpleNamespace(latest_sequence=7, read=read_frame)
        with patch.object(worker, "configure_logging"):
            worker.run_pointcloud_worker(channels, self.config, camera, self.plane)
        published = channels.pointcloud_ring.write.call_args.kwargs
        np.testing.assert_array_equal(published["point_cloud"], block["point_cloud"][0])
        self.assertEqual((published["source_camera_sequence"], published["timestamp_ns"]), (7, 123))
        self.assertFalse(channels.pointcloud_ready.is_set())

        fields = ("eef_pose", "fingertip_xyz", "hand_pointcloud", "point_cloud")
        dependencies = policy_dependencies(fields)
        kinematics = build_observation_kinematics(
            dependencies,
            fingertip_link_names=XHAND_FINGERTIP_LINK_NAMES,
            hand=HandConfig(),
            hand_pointcloud=hand_config,
        )
        rows = [
            PolicyObservationRow(
                123,
                123,
                {"arm_qpos": arm, "hand_qpos": hand, "point_cloud": published["point_cloud"]},
            )
        ]
        online = build_policy_observation(rows, fields, dependencies, kinematics=kinematics)
        for name in fields:
            np.testing.assert_array_equal(online[name], block[name])


if __name__ == "__main__":
    unittest.main()

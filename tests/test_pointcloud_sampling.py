"""Offline checks for point-cloud sampling and equivalent numerical fast paths."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter  # type: ignore[import-untyped]

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_geometry import CameraIntrinsics
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    _deproject_depth,
    _fixed_size_sample,
    _local_3x3_count,
    _packed_grid_keys,
    _ray_plane_factors,
    _reject_flying_depth,
)
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig


class PointCloudSamplingTest(unittest.TestCase):
    def test_downsample_covers_each_coarse_cell_before_hash_fill(self) -> None:
        voxel_keys = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [9, 0, 0],
                [10, 0, 0],
            ],
            dtype=np.int64,
        )
        cloud = np.column_stack(
            (voxel_keys.astype(np.float32), np.zeros((8, 3), dtype=np.float32))
        )

        sampled = _fixed_size_sample(
            cloud,
            voxel_keys,
            num_points=4,
            coarse_voxel_stride=3,
        )

        coarse_x = np.floor_divide(sampled[:, 0].astype(np.int64), 3)
        np.testing.assert_array_equal(np.sort(coarse_x), [0, 1, 2, 3])
        np.testing.assert_array_equal(
            sampled,
            _fixed_size_sample(
                cloud,
                voxel_keys,
                num_points=4,
                coarse_voxel_stride=3,
            ),
        )

    def test_cyclic_padding_reuses_only_valid_candidates(self) -> None:
        voxel_keys = np.asarray([[0, 0, 0], [3, 0, 0]], dtype=np.int64)
        cloud = np.column_stack(
            (voxel_keys.astype(np.float32), np.ones((2, 3), dtype=np.float32))
        )

        sampled = _fixed_size_sample(
            cloud,
            voxel_keys,
            num_points=5,
            coarse_voxel_stride=3,
        )

        self.assertEqual(sampled.shape, (5, 6))
        self.assertTrue(all(np.any(np.all(row == cloud, axis=1)) for row in sampled))

    def test_sampling_policy_and_config_identity_are_explicit(self) -> None:
        config = PointCloudConfig()

        self.assertEqual(config.sampling_coarse_voxel_stride, 3)
        self.assertEqual(config.to_dict()["sampling_coarse_voxel_stride"], 3)
        self.assertTrue(POINT_CLOUD_POLICY_ID.endswith("_v9"))
        self.assertIn("coarse_voxel_stratified", POINT_CLOUD_SAMPLING)


class PointCloudFastPathTest(unittest.TestCase):
    def test_local_count_matches_clipped_3x3_neighborhoods(self) -> None:
        mask = np.asarray(
            [
                [True, False, True, False],
                [False, True, True, False],
                [True, False, False, True],
            ],
            dtype=bool,
        )
        expected = np.empty(mask.shape, dtype=np.uint8)
        for row in range(mask.shape[0]):
            for column in range(mask.shape[1]):
                expected[row, column] = np.count_nonzero(
                    mask[
                        max(0, row - 1) : row + 2,
                        max(0, column - 1) : column + 2,
                    ]
                )

        np.testing.assert_array_equal(_local_3x3_count(mask), expected)

    def test_opencv_depth_neighborhood_matches_scipy_reference(self) -> None:
        rng = np.random.default_rng(20260826)
        config = PointCloudConfig(
            depth_support_min_neighbors=0,
            edge_support_min_neighbors=0,
        )
        for shape in ((1, 7), (7, 1), (8, 11), (37, 53)):
            depth_m = rng.uniform(0.2, 1.0, size=shape).astype(np.float32)
            valid = rng.random(shape) > 0.25
            if shape == (8, 11):
                depth_m[:, 5:] += np.float32(0.1)

            local_min = minimum_filter(
                np.where(valid, depth_m, np.inf),
                size=3,
                mode="constant",
                cval=np.inf,
            )
            local_max = maximum_filter(
                np.where(valid, depth_m, -np.inf),
                size=3,
                mode="constant",
                cval=-np.inf,
            )
            valid_count = np.empty(shape, dtype=np.uint8)
            for row in range(shape[0]):
                for column in range(shape[1]):
                    valid_count[row, column] = np.count_nonzero(
                        valid[
                            max(0, row - 1) : row + 2,
                            max(0, column - 1) : column + 2,
                        ]
                    )

            endpoint_distance = np.minimum(
                depth_m - local_min,
                local_max - depth_m,
            )
            discontinuity = (
                valid
                & (valid_count >= 3)
                & ((local_max - local_min) > config.edge_jump_m)
            )
            expected = valid & ~(
                discontinuity & (endpoint_distance > config.edge_surface_band_m)
            )

            np.testing.assert_array_equal(
                _reject_flying_depth(depth_m, valid, config),
                expected,
            )

    def test_grid_key_packing_is_collision_free_with_negative_keys(self) -> None:
        keys = np.asarray(
            [[-3, 2, 1], [-3, 2, 1], [-3, 2, 2], [4, -5, 1], [4, -5, 2]],
            dtype=np.int64,
        )
        packed = _packed_grid_keys(keys)

        for left in range(len(keys)):
            for right in range(len(keys)):
                self.assertEqual(
                    packed[left] == packed[right],
                    np.array_equal(keys[left], keys[right]),
                )

    def test_grid_key_packing_fails_closed_on_int64_overflow(self) -> None:
        keys = np.asarray(
            [[0, 0, 0], [2_097_152, 2_097_152, 2_097_152]],
            dtype=np.int64,
        )

        with self.assertRaisesRegex(OverflowError, "exceeds int64"):
            _packed_grid_keys(keys)

    def test_cached_rays_preserve_deprojection_and_plane_factors(self) -> None:
        intrinsics = CameraIntrinsics(
            width=4,
            height=3,
            fx=2.0,
            fy=2.5,
            ppx=1.0,
            ppy=1.0,
            distortion_model="none",
            distortion_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        depth_m = np.asarray(
            [
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        )
        valid = np.asarray(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
            dtype=bool,
        )

        points, rows, columns = _deproject_depth(depth_m, valid, intrinsics)
        z = depth_m[rows, columns]
        expected = np.column_stack(
            (
                (columns.astype(np.float32) - intrinsics.ppx) / intrinsics.fx * z,
                (rows.astype(np.float32) - intrinsics.ppy) / intrinsics.fy * z,
                z,
            )
        ).astype(np.float32)
        np.testing.assert_array_equal(points, expected)

        normal = (0.1, -0.2, 0.97)
        factors = _ray_plane_factors(intrinsics, normal)
        np.testing.assert_allclose(
            factors[rows, columns],
            (points / z[:, None]) @ np.asarray(normal),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertFalse(factors.flags.writeable)
        self.assertIs(factors, _ray_plane_factors(intrinsics, normal))


class PointCloudProductionPolicyTest(unittest.TestCase):
    def test_realtime_worker_projects_current_resolved_policy(self) -> None:
        runtime = resolve_runtime_config()
        loop = PointCloudLoopConfig.from_runtime(runtime, num_points=2048)
        expected = runtime.pointcloud.to_dict()
        expected["num_points"] = 2048

        self.assertEqual(loop.pointcloud.to_dict(), expected)
        self.assertEqual(loop.max_input_age_s, runtime.camera.max_frame_age_s)
        self.assertEqual(
            loop.table_plane_abcd,
            (
                runtime.environment.table.plane_abcd
                if runtime.environment.table.enabled
                else None
            ),
        )


if __name__ == "__main__":
    unittest.main()

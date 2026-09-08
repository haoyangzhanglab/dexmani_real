"""Offline characterization for point-cloud diagnostics and pure numerics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import examples.pointcloud_process_example as diagnostic
from dexmani_real.calibration.table import fit_table_plane
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry
from dexmani_real.sensor.pointcloud import (
    PointCloudBuildStats,
    PointCloudBuildTimings,
    build_point_cloud_with_stats,
    build_raw_point_cloud,
)


def _small_geometry() -> RGBDGeometry:
    intrinsics = CameraIntrinsics(
        width=7,
        height=7,
        fx=1000.0,
        fy=1000.0,
        ppx=3.0,
        ppy=3.0,
        distortion_model="none",
        distortion_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    return RGBDGeometry(intrinsics, intrinsics, np.eye(4, dtype=np.float64))


def _synthetic_rgbd() -> tuple[np.ndarray, np.ndarray]:
    depth_raw = np.full((7, 7), 1000, dtype=np.uint16)
    rgb = np.zeros((7, 7, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(7, dtype=np.uint8)[None, :] * 10
    rgb[..., 1] = np.arange(7, dtype=np.uint8)[:, None] * 20
    return depth_raw, rgb


def test_pointcloud_cli_and_canonical_defaults() -> None:
    args = diagnostic._parse_args([])
    assert args.save_dir is None
    assert diagnostic._parse_args(["--save-dir", "snapshots"]).save_dir.name == (
        "snapshots"
    )

    diagnostic_config = diagnostic.PointCloudDiagnosticConfig()
    assert diagnostic_config.rgb_resolution == (640, 480)
    assert diagnostic_config.depth_resolution == (640, 480)
    assert diagnostic_config.fps == 30
    assert diagnostic_config.warmup_frames == 10
    assert diagnostic_config.table_calibration_frames == 5
    assert diagnostic_config.vis_depth_min_m == 0.3
    assert diagnostic_config.vis_depth_max_m == 2.5

    policy = PointCloudConfig()
    assert policy.num_points == 1024
    assert policy.depth_min_m == 0.30
    assert policy.depth_max_m == 1.50
    assert policy.to_dict()["sampling_coarse_voxel_stride"] == 3


def test_synthetic_rgbd_output_is_metric_colored_and_deterministic() -> None:
    geometry = _small_geometry()
    depth_raw, rgb = _synthetic_rgbd()
    identity = np.eye(4, dtype=np.float64)

    raw = build_raw_point_cloud(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=0.001,
        geometry=geometry,
        T_xarm_base_from_color=identity,
    )
    assert raw is not None
    assert raw.shape == (49, 6)
    assert raw.dtype == np.float32
    np.testing.assert_allclose(
        raw[24],
        np.asarray([0.0, 0.0, 1.0, 30 / 255, 60 / 255, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )

    config = PointCloudConfig(
        num_points=16,
        workspace=(-0.1, -0.1, 0.9, 0.1, 0.1, 1.1),
        voxel_size_m=0.0005,
        outlier_radius_m=0.003,
    )
    first, stats = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=0.001,
        geometry=geometry,
        T_xarm_base_from_color=identity,
        table_plane_abcd=None,
        config=config,
    )
    second, _ = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=0.001,
        geometry=geometry,
        T_xarm_base_from_color=identity,
        table_plane_abcd=None,
        config=config,
    )
    assert first is not None
    assert second is not None
    assert first.shape == (16, 6)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.all((first[:, 3:] >= 0.0) & (first[:, 3:] <= 1.0))
    np.testing.assert_array_equal(first, second)
    assert stats.depth_valid_points == 49
    assert stats.depth_trusted_points == 49
    assert stats.cropped_points == 49
    assert stats.voxel_points == 49
    assert stats.spatial_inlier_points == 49


def test_table_fit_is_deterministic_and_recovers_upward_plane() -> None:
    rng = np.random.default_rng(123)
    xy = rng.uniform([-0.5, -0.4], [0.5, 0.4], size=(1000, 2))
    z = (
        0.55
        + 0.025 * xy[:, 0]
        - 0.015 * xy[:, 1]
        + rng.normal(0.0, 0.0005, size=xy.shape[0])
    )
    table_points = np.column_stack((xy, z))
    outliers = rng.uniform(
        [-0.5, -0.4, 0.1], [0.5, 0.4, 1.2], size=(100, 3)
    )
    points = np.concatenate((table_points, outliers), axis=0)
    options = dict(
        distance_threshold_m=0.003,
        max_iterations=300,
        max_evaluated_points=2000,
        min_inlier_points=800,
        min_inlier_ratio=0.5,
        max_tilt_deg=10.0,
        random_seed=42,
    )

    fit = fit_table_plane(points, **options)
    repeat = fit_table_plane(points, **options)

    np.testing.assert_array_equal(fit.plane_abcd, repeat.plane_abcd)
    assert fit.inlier_points == 1000
    assert fit.evaluated_points == 1100
    assert fit.inlier_ratio > 0.9
    assert fit.rms_residual_m < 0.001
    assert fit.max_abs_residual_m < 0.003
    assert fit.plane_abcd[2] > 0.99
    assert abs(fit.plane_abcd[3] + 0.55) < 0.002


def test_benchmark_reports_processing_and_capture_to_cloud_ranges() -> None:
    geometry = _small_geometry()
    config = PointCloudConfig()
    frame = SimpleNamespace(
        rgb=np.zeros((7, 7, 3), dtype=np.uint8),
        depth_aligned_to_color_raw=np.full((7, 7), 1000, dtype=np.uint16),
    )

    class Camera:
        def read(self):
            return frame

        def get_depth_scale(self) -> float:
            return 0.001

    stage_timings = PointCloudBuildTimings(depth_filter_ms=0.1)
    stats = PointCloudBuildStats(candidate_points=1, timings=stage_timings)
    # Each benchmark sample consumes four perf_counter values: frame start,
    # post-capture start, post-build pipeline end, and end-to-end end.
    clock_values = iter(
        [
            0.000,
            0.001,
            0.003,
            0.008,
            0.010,
            0.011,
            0.014,
            0.021,
        ]
    )
    with (
        mock.patch.object(
            diagnostic.time, "perf_counter", side_effect=lambda: next(clock_values)
        ),
        mock.patch.object(
            diagnostic,
            "build_point_cloud_with_stats",
            return_value=(np.ones((1, 6), dtype=np.float32), stats),
        ),
    ):
        _cloud, timings = diagnostic._benchmark_production_pipeline(
            camera=Camera(),
            geometry=geometry,
            T_xarm_base_from_color=np.eye(4),
            config=config,
            table_plane_abcd=None,
            frame_count=2,
        )

    assert timings["pipeline_total"] == pytest.approx(2.5)
    assert timings["pipeline_p50"] == pytest.approx(2.5)
    assert timings["pipeline_p95"] == pytest.approx(2.95)
    assert timings["pipeline_max"] == pytest.approx(3.0)
    assert timings["end_to_end_p50"] == pytest.approx(9.5)
    assert timings["end_to_end_p95"] == pytest.approx(10.85)
    assert timings["end_to_end_max"] == pytest.approx(11.0)
    assert timings["depth_filter_ms_p95"] == pytest.approx(0.1)

"""Offline characterization for point-cloud diagnostics and pure numerics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import examples.pointcloud_process_example as diagnostic
from dexmani_real.calibration.table import fit_table_plane
from dexmani_real.config.experiment import resolve_experiment_config
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
    outliers = rng.uniform([-0.5, -0.4, 0.1], [0.5, 0.4, 1.2], size=(100, 3))
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
        ) as build_cloud,
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
    assert build_cloud.call_count == 2
    for call in build_cloud.call_args_list:
        assert call.kwargs["color"] is frame.rgb
        assert call.kwargs["depth_raw"] is frame.depth_aligned_to_color_raw


def test_reported_cloud_warms_up_then_builds_canonical_output() -> None:
    geometry = _small_geometry()
    depth_raw, rgb = _synthetic_rgbd()
    config = PointCloudConfig()
    stats = PointCloudBuildStats(candidate_points=1)
    expected = np.ones((1, 6), dtype=np.float32)

    with mock.patch.object(
        diagnostic,
        "build_point_cloud_with_stats",
        side_effect=[(None, stats), (expected, stats)],
    ) as build_cloud:
        result = diagnostic._build_cloud(
            depth_raw=depth_raw,
            rgb=rgb,
            depth_scale_m=0.001,
            geometry=geometry,
            T_xarm_base_from_color=np.eye(4),
            config=config,
            table_plane_abcd=None,
        )

    np.testing.assert_array_equal(result, expected)
    assert build_cloud.call_count == 2
    for call in build_cloud.call_args_list:
        assert call.kwargs["color"] is rgb
        assert call.kwargs["depth_raw"] is depth_raw
        assert call.kwargs["table_plane_abcd"] is None


def test_table_calibration_stays_separate_from_production() -> None:
    geometry = _small_geometry()
    depth_raw = np.full((7, 7), 1000, dtype=np.uint16)
    frame = SimpleNamespace(depth_aligned_to_color_raw=depth_raw)
    points = np.asarray(
        [[0.4, 0.0, 0.3], [0.5, 0.1, 0.3], [0.6, -0.1, 0.3]],
        dtype=np.float32,
    )
    fit = SimpleNamespace(
        plane_abcd=(0.0, 0.0, 1.0, -0.3),
        inlier_points=3,
        evaluated_points=3,
        inlier_ratio=1.0,
        rms_residual_m=0.0,
        tilt_deg=0.0,
    )

    class Camera:
        def read(self):
            return frame

        def get_depth_scale(self) -> float:
            return 0.001

    with (
        mock.patch("builtins.input", side_effect=["", "n"]),
        mock.patch.object(
            diagnostic,
            "aligned_depth_points_in_base",
            return_value=points,
        ) as calibration_points,
        mock.patch.object(diagnostic, "fit_table_plane", return_value=fit),
        mock.patch.object(
            diagnostic,
            "build_point_cloud_with_stats",
            side_effect=AssertionError(
                "table calibration must not use production filtering"
            ),
        ),
    ):
        plane, _elapsed_ms = diagnostic._calibrate_table(
            camera=Camera(),
            geometry=geometry,
            T_xarm_base_from_color=np.eye(4),
            config=PointCloudConfig(),
            plane_path=Path("desk_plane.json"),
            frame_count=2,
        )

    assert plane == fit.plane_abcd
    assert calibration_points.call_count == 2
    for call in calibration_points.call_args_list:
        np.testing.assert_array_equal(call.kwargs["depth_raw"], depth_raw)
        assert call.kwargs["aligned_depth_intrinsics"] is geometry.depth


@pytest.mark.parametrize(
    ("calibrate_table", "expected_table_source"),
    [(False, "resolved_runtime"), (True, "calibrated_this_run")],
)
def test_main_uses_post_calibration_capture_for_processing_and_snapshot(
    tmp_path,
    calibrate_table: bool,
    expected_table_source: str,
) -> None:
    runtime = resolve_experiment_config()
    assert runtime.environment.table.enabled
    geometry = _small_geometry()
    preview_rgb = np.zeros((7, 7, 3), dtype=np.uint8)
    preview_depth = np.full((7, 7), 1000, dtype=np.uint16)
    processed_rgb = np.full((7, 7, 3), 17, dtype=np.uint8)
    processed_depth = np.full((7, 7), 1200, dtype=np.uint16)
    processed_cloud = np.ones((runtime.pointcloud.num_points, 6), dtype=np.float32)
    raw_cloud = np.full((2, 6), 0.5, dtype=np.float32)
    calibrated_plane = (0.0, 0.0, 1.0, -0.4)
    events: list[str] = []

    class Camera:
        def __init__(self) -> None:
            self.disconnected = False

        def get_geometry(self):
            return SimpleNamespace(aligned_depth_to_color=lambda: geometry)

        def get_depth_scale(self) -> float:
            return 0.001

        def disconnect(self) -> None:
            self.disconnected = True

    camera = Camera()
    diagnostic_config = diagnostic.PointCloudDiagnosticConfig(
        show_rgbd_panels=False,
        show_o3d=False,
    )

    def capture_frame(_camera):
        if not events:
            events.append("preview_capture")
            return (
                preview_rgb,
                preview_depth,
                preview_depth.astype(np.float32) * 0.001,
                1.0,
            )
        events.append("processed_capture")
        return (
            processed_rgb,
            processed_depth,
            processed_depth.astype(np.float32) * 0.001,
            1.0,
        )

    def calibrate(**_kwargs):
        events.append("calibrate")
        return calibrated_plane, 2.0

    with (
        mock.patch.object(
            diagnostic,
            "PointCloudDiagnosticConfig",
            return_value=diagnostic_config,
        ),
        mock.patch.object(
            diagnostic, "resolve_experiment_config", return_value=runtime
        ),
        mock.patch.object(diagnostic, "_connect_camera", return_value=camera),
        mock.patch.object(diagnostic, "_print_device_info", return_value={}),
        mock.patch.object(
            diagnostic,
            "_capture_frame",
            side_effect=capture_frame,
        ),
        mock.patch.object(diagnostic, "_print_depth_stats"),
        mock.patch.object(diagnostic, "_show_rgbd_panels") as show_rgbd,
        mock.patch.object(
            diagnostic, "_load_extrinsics", return_value=(np.eye(4), 1.0)
        ),
        mock.patch.object(
            diagnostic,
            "_resolve_table_plane_path",
            return_value=tmp_path / "desk_plane.json",
        ),
        mock.patch("builtins.input", return_value="y" if calibrate_table else "n"),
        mock.patch.object(
            diagnostic, "_calibrate_table", side_effect=calibrate
        ) as calibrate_plane,
        mock.patch.object(
            diagnostic, "_build_cloud", return_value=processed_cloud
        ) as build_cloud,
        mock.patch.object(
            diagnostic, "build_raw_point_cloud", return_value=raw_cloud
        ) as build_raw,
        mock.patch.object(
            diagnostic,
            "_save_diagnostic_snapshot",
            return_value=tmp_path / "snapshot",
        ) as save_snapshot,
        mock.patch.object(
            diagnostic,
            "_benchmark_production_pipeline",
            return_value=(np.zeros((0, 6), dtype=np.float32), {"pipeline_total": 1.0}),
        ),
        mock.patch.object(diagnostic, "_print_timing_summary"),
    ):
        assert diagnostic.main(["--save-dir", str(tmp_path)]) == 0

    assert camera.disconnected
    assert events == (
        ["preview_capture", "calibrate", "processed_capture"]
        if calibrate_table
        else ["preview_capture", "processed_capture"]
    )
    assert calibrate_plane.call_count == int(calibrate_table)
    assert show_rgbd.call_args.args[0] is preview_rgb
    assert build_cloud.call_args.kwargs["rgb"] is processed_rgb
    assert build_cloud.call_args.kwargs["depth_raw"] is processed_depth
    expected_plane = (
        calibrated_plane if calibrate_table else runtime.environment.table.plane_abcd
    )
    assert build_cloud.call_args.kwargs["table_plane_abcd"] == expected_plane
    assert build_raw.call_args.kwargs["color"] is processed_rgb
    assert build_raw.call_args.kwargs["depth_raw"] is processed_depth
    assert save_snapshot.call_args.kwargs["rgb"] is processed_rgb
    assert save_snapshot.call_args.kwargs["depth_raw"] is processed_depth
    assert save_snapshot.call_args.kwargs["processed_point_cloud"] is processed_cloud
    assert save_snapshot.call_args.kwargs["raw_point_cloud"] is raw_cloud
    assert save_snapshot.call_args.kwargs["table_plane_abcd"] == expected_plane
    assert save_snapshot.call_args.kwargs["table_plane_source"] == expected_table_source

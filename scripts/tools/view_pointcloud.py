#!/usr/bin/env python3
"""连接 RealSense 查看点云处理效果，支持桌面平面标定及确认后发布。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.calibration.table_plane import (
    publish_table_plane,
    resolve_table_plane,
    resolve_table_plane_path,
)
from dexmani_real.config import load_experiment_config
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.geometry.plane import fit_table_plane
from dexmani_real.perception.pointcloud.config import PointCloudConfig
from dexmani_real.perception.pointcloud.diagnostics import (
    benchmark_production_pipeline,
    build_cloud,
    print_depth_stats,
    save_diagnostic_snapshot,
)
from dexmani_real.perception.pointcloud.processing import (
    aligned_depth_points_in_base,
    build_raw_point_cloud,
)
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.visualization import (
    log_camera_geometry,
    log_points,
    log_rgbd,
    log_workspace,
    viewer_blueprint,
)

if TYPE_CHECKING:
    from dexmani_real.sensor.camera.realsense import RealSenseCamera


@dataclass(frozen=True)
class PointCloudDiagnosticConfig:
    table_calibration_frames: int = 5

    vis_depth_min_m: float = 0.3
    vis_depth_max_m: float = 2.5


SENSOR_OPTION_NAMES = (
    "visual_preset",
    "laser_power",
    "receiver_gain",
    "confidence_threshold",
    "noise_filtering",
    "digital_gain",
    "depth_offset",
    "min_distance",
    "post_processing_sharpening",
    "pre_processing_sharpening",
    "noise_estimation",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense tabletop point-cloud processing and calibration diagnostic"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Atomically save the post-calibration aligned RGB-D frame, raw and "
            "processed clouds, and reconstruction metadata below this directory"
        ),
    )
    parser.add_argument("--config", type=Path, help="Optional experiment YAML overrides")
    parser.add_argument("--camera-calibration", type=Path)
    return parser.parse_args(argv)


def print_device_info(camera: RealSenseCamera) -> dict:
    import pyrealsense2 as rs

    info = camera.get_device_info()
    print("=" * 60)
    print(f"Device:   {info['name']}")
    print(f"Serial:   {info['serial']}")
    print(f"Firmware: {info['firmware']}")
    print("=" * 60)

    device = camera.profile.get_device()
    depth_sensor = device.first_depth_sensor()

    preset = depth_sensor.get_option(rs.option.visual_preset)
    try:
        preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, preset)
    except RuntimeError:
        preset_name = "unknown"
    print(f"  Runtime preset: {preset} {preset_name}")
    print(f"  Depth scale:    {camera.get_depth_scale():.6f} (raw_uint16 x scale = meters)")

    print("\n  --- Sensor read-back ---")
    for name in SENSOR_OPTION_NAMES:
        opt = getattr(rs.option, name)
        try:
            print(f"  {name}: {depth_sensor.get_option(opt)}")
        except RuntimeError:
            print(f"  {name}: <not readable>")
    return info


def capture_frame(
    camera: RealSenseCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture one aligned frame and return RGB and raw/metric depth."""
    print("\nCapturing depth-to-color aligned RGBD frame...")
    frame = camera.read()

    rgb = frame.rgb
    depth_raw = frame.depth_aligned_to_color_raw
    depth_m = frame.depth_aligned_to_color_raw.astype(np.float32) * camera.get_depth_scale()
    if rgb.shape[:2] != depth_raw.shape:
        raise RuntimeError("aligned RGB and depth frame dimensions do not match")

    print(f"  RGB:     shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"  Depth:   shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/{depth_m.size}")
    print(f"  Depth scale: {camera.get_depth_scale():.6f} m")

    return rgb, depth_raw, depth_m


def load_extrinsics(camera_info: dict, calib: CameraExtrinsics) -> np.ndarray:
    """Load T_xarm_base_from_color for aligned depth-to-color samples."""
    print("\nLoading extrinsics from cameras.json...")
    base_from_color = calib.for_serial(camera_info["serial"])

    pos = base_from_color[:3, 3]
    quat_xyzw = R.from_matrix(base_from_color[:3, :3]).as_quat()
    print(
        f"  Camera '{camera_info['name']}' "
        "-> color/aligned-depth frame in xArm-base frame"
    )
    print(f"  pos:         {np.round(pos, 4)} m")
    print(f"  quat (xyzw): {np.round(quat_xyzw, 4)}")
    return base_from_color


def calibrate_table_plane(
    *,
    camera: RealSenseCamera,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    plane_path: Path,
    frame_count: int,
) -> tuple[float, float, float, float]:
    print("\n" + "=" * 60)
    print("Table Plane Calibration (multi-frame RANSAC)")
    print("=" * 60)
    print("  Clear movable objects from the visible table before continuing.")
    input("  Press Enter to capture calibration frames...")

    depth_frames: list[np.ndarray] = []
    for _ in range(frame_count):
        frame = camera.read()
        depth_frames.append(frame.depth_aligned_to_color_raw)

    point_batches: list[np.ndarray] = []
    lower = np.asarray(config.workspace[:3], dtype=np.float32)
    upper = np.asarray(config.workspace[3:], dtype=np.float32)
    for depth_raw in depth_frames:
        points = aligned_depth_points_in_base(
            depth_raw=depth_raw,
            depth_scale_m=camera.get_depth_scale(),
            aligned_depth_intrinsics=intrinsics,
            T_xarm_base_from_color=T_xarm_base_from_color,
            config=config,
        )
        points = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if points.shape[0] > 30_000:
            step = max(1, points.shape[0] // 30_000)
            points = points[::step][:30_000]
        point_batches.append(points)
    calibration_points = np.concatenate(point_batches, axis=0)
    fit = fit_table_plane(calibration_points)

    a, b, c, d = fit.plane_abcd
    print(f"  Plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(
        f"  Support: {fit.inlier_points}/{fit.evaluated_points} "
        f"({fit.inlier_ratio * 100:.1f}%), RMS={fit.rms_residual_m * 1000:.2f} mm, "
        f"tilt={fit.tilt_deg:.2f} deg"
    )

    print(
        "  WARNING: publishing updates the shared plane used by point-cloud "
        "cropping and table collision geometry."
    )
    confirmed = input(f"Publish to {plane_path}? [y/N] ").strip().lower() in {
        "y",
        "yes",
    }
    if confirmed:
        publish_table_plane(plane_path, fit, confirmed=True)
        print("  Published table calibration.")
    else:
        print("  Calibration file unchanged; using the new fit for this run only.")
    return fit.plane_abcd


class PointCloudExample:
    """Own camera diagnostics, optional table fitting and snapshot output."""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        import rerun as rr

        from dexmani_real.sensor.camera.realsense import RealSenseCamera

        self.save_dir = (
            None if self.args.save_dir is None else self.args.save_dir.expanduser().resolve()
        )
        if self.save_dir is not None and self.save_dir.exists() and not self.save_dir.is_dir():
            raise NotADirectoryError(f"snapshot output root is not a directory: {self.save_dir}")
        self.cfg = PointCloudDiagnosticConfig()

        self.fit_table_requested = input("\nRun table calibration? [y/N] ").strip().lower() in {
            "y",
            "yes",
        }
        self.runtime = load_experiment_config(yaml_path=self.args.config)
        self.pcd_config = self.runtime.pointcloud
        self.pcd_config.validate()
        self.calibration = CameraExtrinsics(self.args.camera_calibration)
        self.table_plane_abcd = (
            resolve_table_plane(self.runtime.environment.table)
            if self.pcd_config.remove_table and not self.fit_table_requested
            else None
        )

        configure_logging()
        self.camera = RealSenseCamera(self.runtime.camera)
        print("Connecting to RealSense...")
        try:
            self.camera.connect()
            rr.init("DexMani Point Cloud Diagnostic", spawn=True)
            rr.send_blueprint(viewer_blueprint(camera_in_base=True))
            self.camera_info = print_device_info(self.camera)

            rgb, depth_raw, depth_m = capture_frame(self.camera)
            print_depth_stats(depth_m, self.cfg.vis_depth_min_m, self.cfg.vis_depth_max_m)
            rr.set_time_sequence("capture", 0)
            log_rgbd(
                rgb, depth_raw, depth_scale_m=self.camera.get_depth_scale(),
                depth_range_m=(self.cfg.vis_depth_min_m, self.cfg.vis_depth_max_m),
            )
            input("在 Rerun 查看 RGB-D，按 Enter 继续...")

            self.intrinsics = self.camera.get_color_intrinsics()
            self.T_xarm_base_from_color = load_extrinsics(self.camera_info, self.calibration)
            log_camera_geometry(self.intrinsics, self.T_xarm_base_from_color)
            log_workspace(self.pcd_config.workspace)

            if self.fit_table_requested:
                fitted_plane = calibrate_table_plane(
                    camera=self.camera,
                    intrinsics=self.intrinsics,
                    T_xarm_base_from_color=self.T_xarm_base_from_color,
                    config=self.pcd_config,
                    plane_path=resolve_table_plane_path(self.runtime.environment.table),
                    frame_count=self.cfg.table_calibration_frames,
                )
                self.table_plane_abcd = fitted_plane if self.pcd_config.remove_table else None
                table_plane_source = (
                    "calibrated_this_run" if self.pcd_config.remove_table else "disabled"
                )
            elif self.pcd_config.remove_table:
                table_plane_source = "resolved_runtime"
                print(
                    "  Table calibration skipped; using resolved table_plane.json: "
                    f"{np.round(self.table_plane_abcd, 6)}"
                )
            else:
                self.table_plane_abcd = None
                table_plane_source = "disabled"
                print("  Table calibration skipped; table crop is disabled by runtime config.")

            # Apply the fitted plane to a fresh post-calibration frame.
            rgb, depth_raw, _ = capture_frame(self.camera)
            rr.set_time_sequence("capture", 1)
            log_rgbd(
                rgb, depth_raw, depth_scale_m=self.camera.get_depth_scale(),
                depth_range_m=(self.cfg.vis_depth_min_m, self.cfg.vis_depth_max_m),
            )

            result = build_cloud(
                depth_raw=depth_raw,
                rgb=rgb,
                depth_scale_m=self.camera.get_depth_scale(),
                intrinsics=self.intrinsics,
                T_xarm_base_from_color=self.T_xarm_base_from_color,
                config=self.pcd_config,
                table_plane_abcd=self.table_plane_abcd,
            )
            if self.save_dir is not None:
                raw_cloud = build_raw_point_cloud(
                    depth_raw=depth_raw,
                    color=rgb,
                    depth_scale_m=self.camera.get_depth_scale(),
                    intrinsics=self.intrinsics,
                    T_xarm_base_from_color=self.T_xarm_base_from_color,
                )
                if raw_cloud is None:
                    raw_cloud = np.zeros((0, 6), dtype=np.float32)
                snapshot_path = save_diagnostic_snapshot(
                    output_root=self.save_dir,
                    rgb=rgb,
                    depth_raw=depth_raw,
                    raw_point_cloud=raw_cloud,
                    processed_point_cloud=result,
                    depth_scale_m=self.camera.get_depth_scale(),
                    intrinsics=self.intrinsics,
                    T_xarm_base_from_color=self.T_xarm_base_from_color,
                    table_plane_abcd=self.table_plane_abcd,
                    table_plane_source=table_plane_source,
                    pointcloud_config=self.pcd_config,
                    camera_info=self.camera_info,
                )
                print(f"\nSaved point-cloud diagnostic snapshot: {snapshot_path}")
            benchmark_production_pipeline(
                camera=self.camera,
                intrinsics=self.intrinsics,
                T_xarm_base_from_color=self.T_xarm_base_from_color,
                config=self.pcd_config,
                table_plane_abcd=self.table_plane_abcd,
            )
            log_points("pcd", result)
            input("在 Rerun 查看处理结果，按 Enter 退出...")

            print("\nDone.")

        finally:
            try:
                self.camera.disconnect()
                print("RealSense pipeline stopped cleanly.")
            finally:
                rr.disconnect()

        return 0


def main(argv: list[str] | None = None) -> int:
    return PointCloudExample(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())

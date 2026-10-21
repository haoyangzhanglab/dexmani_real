#!/usr/bin/env python3
"""连接 RealSense 查看点云处理效果，支持桌面平面标定及确认后发布。"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.calibration.table import fit_table_plane, publish_table_plane
from dexmani_real.config.experiment import (
    load_experiment_config,
    resolve_table_plane,
    resolve_table_plane_path,
)
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.pointcloud.processing import (
    aligned_depth_points_in_base,
    build_point_cloud,
    build_raw_point_cloud,
)
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    import open3d as o3d
    from dexmani_real.sensor.camera.realsense import RealSenseCamera


@dataclass(frozen=True)
class PointCloudDiagnosticConfig:
    table_calibration_frames: int = 5

    show_rgbd_panels: bool = True
    show_o3d: bool = True

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


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map normalized [0, 1] values to float RGB [..., 3] in [0, 1]."""
    t = np.clip(values, 0.0, 1.0)
    red = np.where(
        t < 0.375,
        0.0,
        np.where(t < 0.625, 4.0 * t - 1.5, 1.0),
    )
    green = np.where(
        t < 0.125,
        0.0,
        np.where(
            t < 0.375,
            4.0 * t - 0.5,
            np.where(t < 0.625, 1.0, np.where(t < 0.875, 3.5 - 4.0 * t, 0.0)),
        ),
    )
    blue = np.where(
        t < 0.125,
        1.0,
        np.where(t < 0.375, 1.5 - 4.0 * t, 0.0),
    )
    return np.stack([red, green, blue], axis=-1)


def make_depth_vis(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    safe = np.where(valid, depth_m, vmin)
    normalized = (safe - float(vmin)) / (float(vmax) - float(vmin))
    depth_bgr = (jet_colormap(normalized)[..., ::-1] * 255.0).astype(np.uint8)
    depth_bgr[~valid] = 0
    return depth_bgr


def show_rgbd_panels(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cfg: PointCloudDiagnosticConfig,
) -> None:
    if not cfg.show_rgbd_panels:
        return
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    panels: list[tuple[str, np.ndarray]] = [
        ("RGB", bgr),
        ("Depth", make_depth_vis(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)),
    ]
    labeled = []
    for title, image in panels:
        canvas = image.copy()
        cv2.putText(
            canvas,
            title,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(canvas)

    try:
        print("\nShowing RGBD window -- press any key to continue...")
        cv2.imshow("L515 RGBD diagnostic", np.hstack(labeled))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"  RGBD window skipped (no display?): {error}")


def print_depth_stats(depth_m: np.ndarray, vmin: float, vmax: float) -> None:
    print("\n" + "=" * 60)
    print("2-D Depth Gate (diagnostic view)")
    print("=" * 60)
    valid = np.isfinite(depth_m) & (depth_m >= float(vmin)) & (depth_m <= float(vmax))
    ratio = float(np.mean(valid))
    print(
        f"  Range [{vmin:.2f}, {vmax:.2f}] m:  {int(valid.sum()):6d} / "
        f"{depth_m.size} pixels  ({ratio * 100:.1f}%)"
    )


SNAPSHOT_SCHEMA_NAME = "dexmani-real-pointcloud-diagnostic-snapshot"
SNAPSHOT_SCHEMA_VERSION = 3


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


def save_diagnostic_snapshot(
    *,
    output_root: Path,
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    raw_point_cloud: np.ndarray,
    processed_point_cloud: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_plane_source: str,
    pointcloud_config: PointCloudConfig,
    camera_info: dict,
) -> Path:
    """Save pipeline results; RGB-D inputs and calibration are validated upstream."""
    for name, cloud in (
        ("raw_point_cloud", raw_point_cloud),
        ("processed_point_cloud", processed_point_cloud),
    ):
        if not np.all(np.isfinite(cloud)):
            raise ValueError(f"{name} contains NaN/Inf")

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    artifact_name = timestamp.strftime("pointcloud_%Y%m%d_%H%M%S_%f")
    target = root / artifact_name
    temporary = Path(tempfile.mkdtemp(prefix=".pointcloud-tmp-", dir=root))

    arrays = {
        "rgb.npy": rgb,
        "depth_aligned_to_color_raw.npy": depth_raw,
        "raw_point_cloud.npy": raw_point_cloud,
        "processed_point_cloud.npy": processed_point_cloud,
        "T_xarm_base_from_color.npy": T_xarm_base_from_color,
    }
    metadata = {
        "schema_name": SNAPSHOT_SCHEMA_NAME,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": timestamp.isoformat(),
        "payload_mode": "depth_to_color_aligned_rgbd",
        "depth_payload_frame": "color",
        "depth_scale_m_per_unit": float(depth_scale_m),
        "color_intrinsics": intrinsics.to_dict(),
        "table_plane_abcd": None if table_plane_abcd is None else list(table_plane_abcd),
        "table_plane_source": table_plane_source,
        "pointcloud_config": asdict(pointcloud_config),
        "point_cloud_frame": "xarm_base",
        "point_cloud_columns": ["x_m", "y_m", "z_m", "r", "g", "b"],
        "raw_point_cloud_semantics": "all_finite_nonzero_aligned_depth_pixels",
        "processed_point_cloud_empty": processed_point_cloud.shape[0] == 0,
        "camera": {key: camera_info[key] for key in ("name", "serial", "firmware")},
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }

    try:
        for name, value in arrays.items():
            np.save(temporary / name, value, allow_pickle=False)
        atomic_json_dump(metadata, temporary / "metadata.json", ensure_ascii=False)
        return atomic_publish(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


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
        f"  Camera '{camera_info.get('name', camera_info['serial'])}' "
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


def build_cloud(
    *,
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    table_plane_abcd: tuple[float, float, float, float] | None,
) -> np.ndarray:
    print("\n" + "=" * 60)
    print("Point-Cloud Pipeline (depth_to_color -> xArm-base)")
    print("=" * 60)
    table_state = "ENABLED" if config.remove_table else "DISABLED"
    print(f"  num_points={config.num_points}  workspace={config.workspace}")
    print(f"  depth range [{config.depth_min_m}, {config.depth_max_m}] m")
    print(
        "  3x3 support neighbors "
        f"flat={config.depth_support_min_neighbors} "
        f"edge={config.edge_support_min_neighbors}"
    )
    print(
        "  radius-component minimum "
        f"{config.outlier_min_component_points} points; radius neighbors "
        f"{config.outlier_min_neighbors}"
    )
    if config.remove_table:
        print(
            "  table height hysteresis "
            f"core={config.table_core_height_m * 1000.0:.1f} mm "
            f"object_seed={config.table_object_seed_height_m * 1000.0:.1f} mm "
            f"x{config.table_object_seed_min_pixels} pixels"
        )
    print(f"  voxel_size_m={config.voxel_size_m}  table crop: {table_state}")

    t0 = time.perf_counter()
    result = build_point_cloud(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        intrinsics=intrinsics,
        T_xarm_base_from_color=T_xarm_base_from_color,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if result is not None:
        print(f"\n  Output: {result.shape[0]} points  ({elapsed_ms:.1f} ms)")
    else:
        print(f"\n  Output: none ({elapsed_ms:.1f} ms)")
        result = np.zeros((0, 6), dtype=np.float32)

    return result


def benchmark_production_pipeline(
    *,
    camera: RealSenseCamera,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    table_plane_abcd: tuple[float, float, float, float] | None,
    frame_count: int = 20,
) -> None:
    elapsed_ms: list[float] = []
    end_to_end_ms: list[float] = []
    valid_clouds = 0
    for _ in range(frame_count):
        frame_started = time.perf_counter()
        frame = camera.read()
        started = time.perf_counter()
        cloud = build_point_cloud(
            depth_raw=frame.depth_aligned_to_color_raw,
            color=frame.rgb,
            depth_scale_m=camera.get_depth_scale(),
            intrinsics=intrinsics,
            T_xarm_base_from_color=T_xarm_base_from_color,
            table_plane_abcd=table_plane_abcd,
            config=config,
        )
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        end_to_end_ms.append((time.perf_counter() - frame_started) * 1000.0)
        valid_clouds += cloud is not None
    values = np.asarray(elapsed_ms, dtype=np.float64)
    end_to_end_values = np.asarray(end_to_end_ms, dtype=np.float64)
    print(
        f"  Fresh-frame benchmark: n={frame_count} "
        f"p50={np.percentile(values, 50):.1f} ms "
        f"p95={np.percentile(values, 95):.1f} ms; valid clouds={valid_clouds}/{frame_count}"
    )
    print(
        f"  Capture-to-cloud benchmark: p50={np.percentile(end_to_end_values, 50):.1f} ms "
        f"p95={np.percentile(end_to_end_values, 95):.1f} ms"
    )


def build_workspace_box(workspace: tuple[float, ...]) -> "o3d.geometry.LineSet":
    import open3d as o3d

    ws = workspace
    corners = np.array(
        [
            [ws[0], ws[1], ws[2]],
            [ws[3], ws[1], ws[2]],
            [ws[3], ws[4], ws[2]],
            [ws[0], ws[4], ws[2]],
            [ws[0], ws[1], ws[5]],
            [ws[3], ws[1], ws[5]],
            [ws[3], ws[4], ws[5]],
            [ws[0], ws[4], ws[5]],
        ]
    )
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(corners)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in edges])
    return box


def visualize_result(
    result: np.ndarray,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
) -> None:
    import open3d as o3d

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    color_camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    color_camera_frame.transform(T_xarm_base_from_color)

    crop_box = build_workspace_box(config.workspace)

    pcd = o3d.geometry.PointCloud()
    if result.shape[0]:
        pcd.points = o3d.utility.Vector3dVector(result[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(result[:, 3:6].astype(np.float64))

    print("\nLaunching open3d visualizer (close window to continue)...")
    print("  xArm-base point cloud + workspace crop box (green) + coordinate frames")
    o3d.visualization.draw_geometries(
        [pcd, crop_box, base_frame, color_camera_frame],
        window_name="Point Cloud",
        point_show_normal=False,
    )


class PointCloudExample:
    """Own camera diagnostics, optional table fitting and snapshot output."""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
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
        self.camera.connect()

        try:
            self.camera_info = print_device_info(self.camera)

            rgb, _, depth_m = capture_frame(self.camera)
            print_depth_stats(depth_m, self.cfg.vis_depth_min_m, self.cfg.vis_depth_max_m)
            show_rgbd_panels(rgb, depth_m, self.cfg)

            self.intrinsics = self.camera.get_color_intrinsics()
            self.T_xarm_base_from_color = load_extrinsics(self.camera_info, self.calibration)

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
            if self.cfg.show_o3d:
                visualize_result(result, self.T_xarm_base_from_color, self.pcd_config)
            else:
                print("\n  open3d visualizer skipped (show_o3d=False)")

            print("\nDone.")

        finally:
            self.camera.disconnect()
            print("RealSense pipeline stopped cleanly.")

        return 0


def main(argv: list[str] | None = None) -> int:
    return PointCloudExample(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())

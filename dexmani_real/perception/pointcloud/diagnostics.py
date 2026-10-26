"""点云处理诊断、快照产物和采集性能测量；相机由调用方持有。"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.perception.pointcloud.config import PointCloudConfig
from dexmani_real.perception.pointcloud.processing import build_point_cloud
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish

if TYPE_CHECKING:
    from dexmani_real.sensor.camera.realsense import RealSenseCamera


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

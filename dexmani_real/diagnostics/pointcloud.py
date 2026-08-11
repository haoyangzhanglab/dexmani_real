"""Inspect the production point-cloud pipeline or fit a desk-plane candidate.

Desk calibration is report-only by default. Updating the operational JSON
requires ``calibrate-desk --write-calibration`` and two operator confirmations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DESK_QUALITY_Z_BAND_MARGIN_M = 0.05


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {text!r}")
    return value


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite, got {text!r}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text!r}")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {text!r}")
    return value


@dataclass(frozen=True)
class DeskPlaneQuality:
    accepted: bool
    sample_count: int
    inlier_ratio: float
    tilt_deg: float
    desk_z_m: float
    median_residual_m: float
    reason: str
    quality_z_band_margin_m: float = _DESK_QUALITY_Z_BAND_MARGIN_M


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production L515 point-cloud diagnostics")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("inspect", "calibrate-desk"),
        default="inspect",
        help="inspect one production frame or fit a desk-plane candidate",
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--serial", default=None, help="Camera serial override")
    parser.add_argument("--timeout-ms", type=_positive_int, default=5000, help="Frame read timeout")
    parser.add_argument(
        "--write-calibration",
        action="store_true",
        help="After quality checks and confirmation, replace the operational desk_plane.json",
    )
    parser.add_argument("--max-tilt-deg", type=_positive_float, default=10.0)
    parser.add_argument("--desk-z-min-m", type=_finite_float, default=-0.05)
    parser.add_argument("--desk-z-max-m", type=_finite_float, default=0.15)
    parser.add_argument("--min-inlier-ratio", type=_fraction, default=0.60)
    parser.add_argument("--inlier-threshold-m", type=_positive_float, default=0.01)
    parser.add_argument("--min-quality-points", type=_positive_int, default=1000)
    return parser


def _validate_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("T_world_camera must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("T_world_camera has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("T_world_camera rotation is not orthonormal")
    return value


def _load_runtime_and_camera(args: argparse.Namespace) -> tuple[Any, Any]:
    from dexmani_real.config.runtime import resolve_runtime_config
    from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

    runtime = resolve_runtime_config(
        yaml_path=args.config,
        cli_overrides={"camera.serial": args.serial},
    )
    camera_cfg = runtime.camera
    camera = RealSense(
        RealSenseConfig(
            serial=camera_cfg.serial,
            depth_resolution=(int(camera_cfg.width), int(camera_cfg.height)),
            color_resolution=(int(camera_cfg.width), int(camera_cfg.height)),
            fps=int(camera_cfg.fps),
            align_mode=cast(Literal["depth_to_color", "color_to_depth", "none"], str(camera_cfg.align_mode)),
            warmup_frames=int(camera_cfg.warmup_frames),
        )
    )
    return runtime, camera


def _resolve_extrinsics(camera: Any) -> tuple[np.ndarray, str]:
    from dexmani_real.config.camera_calib import CameraCalib

    serial = str(camera.get_device_info().get("serial", ""))
    if not serial:
        raise RuntimeError("connected camera did not report a serial number")
    calibration = CameraCalib()
    camera_name = calibration.resolve_name_by_serial(serial)
    transform = _validate_transform(calibration.get_extrinsics(camera_name))
    return transform, camera_name


def _capture_rgbd(camera: Any, timeout_ms: int) -> Any:
    frame = camera.read(timeout_ms=timeout_ms)
    if frame.rgb is None:
        raise RuntimeError("RGB is required by the production point-cloud pipeline")
    if frame.align_mode == "none":
        raise RuntimeError("production point-cloud diagnostics require aligned RGB-D")
    depth = np.asarray(frame.depth)
    rgb = np.asarray(frame.rgb)
    rays = np.asarray(camera.get_rays())
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or rays.shape != (*depth.shape, 3):
        raise RuntimeError(f"unaligned RGB-D/rays: depth={depth.shape}, rgb={rgb.shape}, rays={rays.shape}")
    return frame


def evaluate_desk_plane_quality(
    depth_m: np.ndarray,
    rays: np.ndarray,
    transform: np.ndarray,
    plane: Sequence[float],
    *,
    depth_min_m: float,
    depth_max_m: float,
    workspace: Sequence[float],
    desk_z_min_m: float,
    desk_z_max_m: float,
    max_tilt_deg: float,
    inlier_threshold_m: float,
    min_inlier_ratio: float,
    min_quality_points: int,
) -> DeskPlaneQuality:
    """Validate a fitted plane against plausible tabletop geometry."""
    coefficients = np.asarray(plane, dtype=np.float64)
    if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
        return DeskPlaneQuality(False, 0, 0.0, float("nan"), float("nan"), float("nan"), "nonfinite plane")
    normal = coefficients[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return DeskPlaneQuality(False, 0, 0.0, float("nan"), float("nan"), float("nan"), "zero normal")
    coefficients = coefficients / norm
    if coefficients[2] < 0.0:
        coefficients = -coefficients
    tilt_deg = float(np.degrees(np.arccos(np.clip(coefficients[2], -1.0, 1.0))))
    if abs(float(coefficients[2])) <= 1e-9:
        desk_z_m = float("inf")
    else:
        desk_z_m = float(-coefficients[3] / coefficients[2])

    depth = np.asarray(depth_m, dtype=np.float64)
    ray_array = np.asarray(rays, dtype=np.float64)
    if depth.ndim != 2 or ray_array.shape != (*depth.shape, 3):
        return DeskPlaneQuality(False, 0, 0.0, tilt_deg, desk_z_m, float("nan"), "invalid depth/rays")
    valid = np.isfinite(depth) & (depth > depth_min_m) & (depth < depth_max_m)
    camera_points = ray_array[valid] * depth[valid, None]
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    x_min, y_min, _z_min, x_max, y_max, _z_max = [float(value) for value in workspace]
    quality_band = (
        (world_points[:, 0] >= x_min)
        & (world_points[:, 0] <= x_max)
        & (world_points[:, 1] >= y_min)
        & (world_points[:, 1] <= y_max)
        & (world_points[:, 2] >= desk_z_min_m - _DESK_QUALITY_Z_BAND_MARGIN_M)
        & (world_points[:, 2] <= desk_z_max_m + _DESK_QUALITY_Z_BAND_MARGIN_M)
    )
    world_points = world_points[quality_band]
    sample_count = int(world_points.shape[0])
    if sample_count:
        residuals = np.abs(world_points @ coefficients[:3] + coefficients[3])
        inlier_ratio = float(np.mean(residuals <= inlier_threshold_m))
        median_residual_m = float(np.median(residuals))
    else:
        inlier_ratio = 0.0
        median_residual_m = float("inf")

    reasons: list[str] = []
    if sample_count < min_quality_points:
        reasons.append(f"only {sample_count} quality points")
    if tilt_deg > max_tilt_deg:
        reasons.append(f"tilt {tilt_deg:.2f}deg exceeds {max_tilt_deg:.2f}deg")
    if not desk_z_min_m <= desk_z_m <= desk_z_max_m:
        reasons.append(f"desk z {desk_z_m:.4f}m outside [{desk_z_min_m:.4f}, {desk_z_max_m:.4f}]m")
    if inlier_ratio < min_inlier_ratio:
        reasons.append(f"inlier ratio {inlier_ratio:.3f} below {min_inlier_ratio:.3f}")
    return DeskPlaneQuality(
        accepted=not reasons,
        sample_count=sample_count,
        inlier_ratio=inlier_ratio,
        tilt_deg=tilt_deg,
        desk_z_m=desk_z_m,
        median_residual_m=median_residual_m,
        reason="; ".join(reasons) if reasons else "accepted",
    )


def _write_plane_atomic(path: Path, plane: Sequence[float]) -> Path | None:
    """Back up the existing calibration and atomically replace it."""
    coefficients = np.asarray(plane, dtype=np.float64)
    if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("desk plane must contain four finite coefficients")
    payload = dict(zip(("a", "b", "c", "d"), coefficients.tolist()))

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    backup: Path | None = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            suffix = f".bak.{time.strftime('%Y%m%dT%H%M%S')}.{time.time_ns()}"
            backup = path.with_name(path.name + suffix)
            shutil.copy2(path, backup)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return backup


def _inspect(runtime: Any, camera: Any, transform: np.ndarray, timeout_ms: int) -> int:
    from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig

    frame = _capture_rgbd(camera, timeout_ms)
    config = PointCloudProcessorConfig(num_points=int(runtime.camera.pointcloud_num_points))
    processor = PointCloudProcessor(transform, config)
    started_s = time.monotonic()
    result = processor.process(frame.depth, frame.rgb, camera.get_rays())
    elapsed_ms = (time.monotonic() - started_s) * 1000.0
    if result is None:
        print(
            f"No point cloud survived production gates; valid_depth={processor.last_valid_depth_ratio:.3f} "
            f"elapsed={elapsed_ms:.1f}ms"
        )
        return 1
    print(
        f"pointcloud={result.shape} source_points={processor.last_source_point_count} "
        f"padding={processor.last_padding_count} valid_depth={processor.last_valid_depth_ratio:.3f} "
        f"elapsed={elapsed_ms:.1f}ms"
    )
    return 0


def _calibrate(args: argparse.Namespace, camera: Any, transform: np.ndarray) -> int:
    from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig

    config = PointCloudProcessorConfig()
    print("Clear the complete tabletop and keep the robot stationary.")
    if input("Type READY to capture a new calibration frame, or anything else to cancel: ").strip() != "READY":
        print("Calibration cancelled before capture.")
        return 2

    frame = _capture_rgbd(camera, args.timeout_ms)
    plane = PointCloudProcessor.calibrate_desk_plane(frame.depth, frame.rgb, camera.get_rays(), transform)
    quality = evaluate_desk_plane_quality(
        frame.depth,
        camera.get_rays(),
        transform,
        plane,
        depth_min_m=float(config.depth_min_m),
        depth_max_m=float(config.depth_max_m),
        workspace=config.workspace,
        desk_z_min_m=args.desk_z_min_m,
        desk_z_max_m=args.desk_z_max_m,
        max_tilt_deg=args.max_tilt_deg,
        inlier_threshold_m=args.inlier_threshold_m,
        min_inlier_ratio=args.min_inlier_ratio,
        min_quality_points=args.min_quality_points,
    )
    print(json.dumps({"plane": list(plane), "quality": quality.__dict__}, indent=2))
    if not quality.accepted:
        print(f"Rejected: {quality.reason}")
        return 1
    if not args.write_calibration:
        print("Candidate accepted; operational calibration was not changed.")
        return 0
    if input("Type WRITE to replace dexmani_real/config/desk_plane.json: ").strip() != "WRITE":
        print("Accepted candidate was not written.")
        return 2

    output_path = (_REPO_ROOT / config.desk_plane_path).resolve()
    backup = _write_plane_atomic(output_path, plane)
    print(f"Wrote {output_path}")
    if backup is not None:
        print(f"Backup: {backup}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.serial is not None and not args.serial.strip():
        raise SystemExit("--serial must be non-empty")
    if args.desk_z_min_m >= args.desk_z_max_m:
        raise SystemExit("--desk-z-min-m must be less than --desk-z-max-m")
    if args.write_calibration and args.mode != "calibrate-desk":
        raise SystemExit("--write-calibration is valid only with calibrate-desk")

    camera: Any | None = None
    try:
        runtime, camera = _load_runtime_and_camera(args)
        if not camera.connect():
            raise RuntimeError("RealSense connect failed")
        transform, camera_name = _resolve_extrinsics(camera)
        print(f"camera={camera_name} serial={camera.get_device_info().get('serial', '')}")
        if args.mode == "calibrate-desk":
            return _calibrate(args, camera, transform)
        return _inspect(runtime, camera, transform, args.timeout_ms)
    except (EOFError, KeyboardInterrupt):
        print("Cancelled.")
        return 130
    except Exception:
        logger.error("Point-cloud diagnostic failed", exc_info=True)
        return 1
    finally:
        if camera is not None:
            camera.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

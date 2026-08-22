#!/usr/bin/env python3
"""Offline native RGB-D point-cloud shadow check for one L515 capture.

The input directory must be created by ``examples/inspect_l515.py --save-rgb``.
This program opens no hardware and does not modify calibration.  It verifies
the recorded native payload and runs the production ``build_point_cloud`` once
per captured frame using the active calibration and table configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_SAMPLING,
    PointCloudConfig,
    build_point_cloud,
)
from dexmani_real.utils.schema import (
    SUPPORTED_POINT_CLOUD_COUNTS,
    validate_point_cloud_array,
)


def _percentiles_ms(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native RGB-D point-cloud shadow path on an L515 capture."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="inspect_l515 output directory containing report.json and native arrays",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional positive cap for a fast diagnostic; default checks every frame.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        choices=sorted(SUPPORTED_POINT_CLOUD_COUNTS),
        default=1024,
        help="Fixed output point count N (default: 1024).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON result path; default is "
            "<input_dir>/native_pointcloud_shadow_<N>.json."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def _read_capture(input_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    report_path = input_dir / "report.json"
    depth_path = input_dir / "native_depth_z16.npy"
    color_path = input_dir / "native_color_rgb8.npy"
    missing = [
        str(path)
        for path in (report_path, depth_path, color_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "native RGB shadow input is incomplete; recapture with "
            f"inspect_l515.py --save-rgb. Missing: {missing}"
        )
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("report.json must contain an object")
    depth = np.load(depth_path, mmap_mode="r")
    color = np.load(color_path, mmap_mode="r")
    if (
        depth.dtype != np.uint16
        or depth.ndim != 3
        or color.dtype != np.uint8
        or color.ndim != 4
        or color.shape[-1] != 3
        or depth.shape[0] != color.shape[0]
    ):
        raise ValueError(
            "native arrays must be depth uint16[F,H,W] and color uint8[F,H,W,3]"
        )
    return report, depth, color


def _geometry_from_report(report: dict[str, Any]) -> RGBDGeometry:
    depth = report.get("depth_stream")
    color = report.get("color_stream")
    if not isinstance(depth, dict) or not isinstance(color, dict):
        raise ValueError("report is missing depth_stream or color_stream")
    depth_intrinsics = depth.get("intrinsics")
    color_intrinsics = color.get("intrinsics")
    transform = report.get("T_color_from_depth")
    if not isinstance(depth_intrinsics, dict) or not isinstance(color_intrinsics, dict):
        raise ValueError("report is missing stream intrinsics")
    return RGBDGeometry.from_dict(
        {
            "depth": depth_intrinsics,
            "color": color_intrinsics,
            "T_color_from_depth": transform,
        }
    )


def _select_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames >= frame_count:
        return np.arange(frame_count, dtype=np.int64)
    return (np.arange(max_frames, dtype=np.int64) * frame_count) // max_frames


def _base_from_depth(report: dict[str, Any], geometry: RGBDGeometry) -> np.ndarray:
    serial = report.get("serial")
    if not isinstance(serial, str) or not serial:
        raise ValueError("report serial must be a non-empty string")
    calibration = CameraCalib()
    camera_name = calibration.resolve_name_by_serial(serial)
    calibration_meta = calibration.to_meta_dict(camera_name, expected_serial=serial)
    if calibration_meta["camera_type"] != "eye_to_hand":
        raise ValueError("shadow check requires an eye_to_hand camera calibration")
    base_from_color = calibration.get_extrinsics(camera_name)
    return base_from_color @ geometry.T_color_from_depth


def run_shadow_check(
    input_dir: Path,
    *,
    max_frames: int | None = None,
    num_points: int = 1024,
) -> dict[str, Any]:
    """Return one deterministic shadow result without opening hardware."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    if isinstance(num_points, bool) or num_points not in SUPPORTED_POINT_CLOUD_COUNTS:
        raise ValueError(
            f"num_points must be one of {sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
        )
    report, depth, color = _read_capture(input_dir)
    geometry = _geometry_from_report(report)
    if depth.shape[1:] != (geometry.depth.height, geometry.depth.width):
        raise ValueError(
            "native depth array shape does not match reported depth intrinsics"
        )
    if color.shape[1:] != (
        geometry.color.height,
        geometry.color.width,
        3,
    ):
        raise ValueError(
            "native color array shape does not match reported color intrinsics"
        )
    depth_scale_m = float(report.get("depth_scale_m", np.nan))
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("report depth_scale_m must be finite and positive")

    runtime = resolve_runtime_config()
    table = runtime.environment.table
    base_from_depth = _base_from_depth(report, geometry)
    indices = _select_indices(depth.shape[0], max_frames)
    config = PointCloudConfig(num_points=num_points)
    durations_ms: list[float] = []
    failed_indices: list[int] = []
    color_mean: list[float] = []
    for index in indices:
        started_ns = time.perf_counter_ns()
        cloud = build_point_cloud(
            depth_raw=np.asarray(depth[index]),
            color=np.asarray(color[index]),
            depth_scale_m=depth_scale_m,
            geometry=geometry,
            T_xarm_base_from_depth=base_from_depth,
            table_plane_abcd=table.plane_abcd if table.enabled else None,
            config=config,
        )
        durations_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
        if cloud is None:
            failed_indices.append(int(index))
            continue
        validate_point_cloud_array(
            cloud,
            num_points=config.num_points,
            label="build_point_cloud output",
        )
        color_mean.append(float(np.mean(cloud[:, 3:])))

    passed = not failed_indices
    return {
        "input_dir": str(input_dir.resolve()),
        "serial": report["serial"],
        "source_scene": report.get("scene"),
        "frames_checked": int(indices.size),
        "frame_indices": indices.tolist(),
        "pointcloud_config": {
            **config.to_dict(),
            "sampling": POINT_CLOUD_SAMPLING,
            "color_source": POINT_CLOUD_COLOR_SOURCE,
        },
        "pointcloud_available_ratio": float(
            (indices.size - len(failed_indices)) / max(1, indices.size)
        ),
        "failed_frame_indices": failed_indices,
        "build_point_cloud_ms": _percentiles_ms(durations_ms),
        "cloud_rgb_mean": _percentiles_ms(color_mean),
        "shadow_gate_passed": passed,
        "gate_explanation": (
            "Every checked native RGB-D frame produced finite fixed-size xArm-base "
            "clouds with projected RGB in [0,1]."
            if passed
            else "At least one checked frame produced no in-workspace visible cloud."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_shadow_check(
            args.input_dir,
            max_frames=args.max_frames,
            num_points=args.num_points,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"Native point-cloud shadow check failed: {exc}", file=sys.stderr)
        return 1
    output = args.output or (
        args.input_dir / f"native_pointcloud_shadow_{args.num_points}.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0 if result["shadow_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

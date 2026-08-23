#!/usr/bin/env python3
"""Inspect one L515 and optionally capture a native-depth benchmark.

This diagnostic connects only to a RealSense camera. It does not connect to or
command the robot, open a GUI, or write calibration. When ``--output-dir`` is
provided it writes one JSON report and native ``.npy`` frame arrays for the
named scene, plus per-frame numbers and timestamps in ``frame_timing.npz``.
It is read-only unless ``--visual-preset`` is explicitly supplied for a
diagnostic preset ablation; that setting is volatile and applies only through
the SDK.

This script's acquisition loop includes geometry and depth-quality
computation. For precise stream-cadence / RGB-FPS root-cause diagnosis use
``diagnose_l515_rgb_timing.py``.

Example::

    python examples/inspect_l515.py \
        --scene inclined_plane \
        --frames 300 \
        --plane-roi 160,120,320,240 \
        --output-dir l515_baseline/inclined_plane
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrealsense2 as rs  # type: ignore[import-not-found]
from scipy.ndimage import (  # type: ignore[import-untyped]
    convolve,
    maximum_filter,
    minimum_filter,
)

_SCENES = (
    "flat_matte_plane",
    "hard_depth_edge",
    "thin_foreground_object",
    "dark_object",
    "bright_reflective_object",
    "inclined_plane",
)
_OPTION_NAMES = (
    "visual_preset",
    "confidence_threshold",
    "laser_power",
    "receiver_gain",
    "noise_filtering",
    "zero_order_enabled",
    "invalidation_bypass",
)


@dataclass(frozen=True)
class Region:
    """One image region in ``x,y,width,height`` coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def rows(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def columns(self) -> slice:
        return slice(self.x, self.x + self.width)

    def validate(self, *, image_width: int, image_height: int) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid non-positive ROI: {self}")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(
                f"ROI {self} exceeds image bounds {image_width}x{image_height}"
            )

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def _parse_region(value: str) -> Region:
    try:
        x, y, width, height = (int(item.strip()) for item in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    return Region(x=x, y=y, width=width, height=height)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one L515's native RGB-D geometry, timing, and depth baseline."
    )
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--visual-preset",
        type=int,
        default=None,
        help="Diagnostic only: apply this volatile base preset before capture.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        choices=range(4),
        default=None,
        help="Diagnostic only: override confidence after --visual-preset.",
    )
    parser.add_argument("--scene", choices=_SCENES, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="Also save native RGB frames (large: about 276 MB for 300 frames).",
    )
    parser.add_argument(
        "--plane-roi",
        type=_parse_region,
        default=None,
        help="Optional x,y,width,height depth ROI used for SDK-deprojected PlaneRMS.",
    )
    parser.add_argument(
        "--foreground-roi",
        type=_parse_region,
        default=None,
        help="Optional x,y,width,height ROI for foreground nonzero retention.",
    )
    parser.add_argument(
        "--background-roi",
        type=_parse_region,
        default=None,
        help="Optional x,y,width,height ROI for background nonzero retention.",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if args.warmup_frames < 0 or args.frames <= 0 or args.timeout_ms <= 0:
        parser.error("warmup must be non-negative; frames and timeout must be positive")
    if args.output_dir is not None and args.scene is None:
        parser.error("--scene is required when --output-dir is used")
    if args.visual_preset is not None and args.visual_preset < 0:
        parser.error("--visual-preset must be non-negative")
    if args.confidence_threshold is not None and args.visual_preset is None:
        parser.error("--confidence-threshold requires --visual-preset")
    for region in (args.plane_roi, args.foreground_roi, args.background_roi):
        if region is not None:
            try:
                region.validate(image_width=args.width, image_height=args.height)
            except ValueError as exc:
                parser.error(str(exc))
    return args


def _device_info(device: Any, key: Any) -> str:
    if key is None:
        return ""
    try:
        return str(device.get_info(key)) if device.supports(key) else ""
    except RuntimeError:
        return ""


def _select_serial(context: Any, requested: str | None) -> str:
    devices = context.query_devices()
    serials = [_device_info(device, rs.camera_info.serial_number) for device in devices]
    serials = [serial for serial in serials if serial]
    if requested is not None:
        if requested not in serials:
            raise RuntimeError(
                f"configured serial {requested!r} is not connected; connected={serials}"
            )
        return requested
    if len(serials) != 1:
        raise RuntimeError(
            "connect exactly one RealSense or pass --serial; " f"connected={serials}"
        )
    return serials[0]


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "distortion_coeffs": [float(value) for value in intrinsics.coeffs],
    }


def _extrinsics_matrix(extrinsics: Any) -> np.ndarray:
    """Convert SDK column-major rotation and verify it against the SDK oracle."""
    raw_rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsics.translation, dtype=np.float64)
    oracle_inputs = np.vstack((np.zeros(3), np.eye(3)))
    oracle = np.asarray(
        [
            rs.rs2_transform_point_to_point(extrinsics, point.tolist())
            for point in oracle_inputs
        ],
        dtype=np.float64,
    )
    for rotation in (raw_rotation.T, raw_rotation):
        transformed = oracle_inputs @ rotation.T + translation
        if np.allclose(transformed, oracle, rtol=0.0, atol=1e-7):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            return matrix
    raise RuntimeError("failed to decode RealSense extrinsic rotation convention")


def _option_snapshot(sensor: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _OPTION_NAMES:
        option = getattr(rs.option, name, None)
        if option is None:
            result[name] = {"exposed": False}
            continue
        try:
            supported = bool(sensor.supports(option))
        except RuntimeError:
            supported = False
        if not supported:
            result[name] = {"exposed": True, "supported": False}
            continue
        item: dict[str, Any] = {
            "exposed": True,
            "supported": True,
            "value": float(sensor.get_option(option)),
        }
        option_range = None
        try:
            option_range = sensor.get_option_range(option)
            item["range"] = {
                "min": float(option_range.min),
                "max": float(option_range.max),
                "step": float(option_range.step),
                "default": float(option_range.default),
            }
        except RuntimeError:
            pass
        if name == "visual_preset" and option_range is not None:
            values = np.arange(
                option_range.min,
                option_range.max + 0.5 * option_range.step,
                option_range.step,
            )
            descriptions: dict[str, str] = {}
            if values.size <= 32:
                for value in values:
                    try:
                        description = sensor.get_option_value_description(
                            option, float(value)
                        )
                    except RuntimeError:
                        continue
                    if description:
                        descriptions[str(float(value))] = str(description)
            item["value_descriptions"] = descriptions
        try:
            description = sensor.get_option_value_description(option, item["value"])
            if description:
                item["value_description"] = str(description)
        except RuntimeError:
            pass
        result[name] = item
    return result


def _apply_setting_variant(
    sensor: Any,
    *,
    visual_preset: int | None,
    confidence_threshold: int | None,
) -> dict[str, Any] | None:
    """Apply an explicitly requested diagnostic variant and record readbacks."""
    if visual_preset is None:
        return None
    if not sensor.supports(rs.option.visual_preset):
        raise RuntimeError("connected L515 does not expose visual_preset")
    sensor.set_option(rs.option.visual_preset, float(visual_preset))
    time.sleep(0.5)
    base_readbacks = _option_snapshot(sensor)
    actual_preset = base_readbacks["visual_preset"].get("value")
    if actual_preset is None or not np.isclose(
        float(actual_preset), float(visual_preset), atol=1e-6
    ):
        raise RuntimeError(
            "L515 visual preset readback mismatch: "
            f"requested={visual_preset}, actual={actual_preset}"
        )

    if confidence_threshold is not None:
        if not sensor.supports(rs.option.confidence_threshold):
            raise RuntimeError("connected L515 does not expose confidence_threshold")
        sensor.set_option(rs.option.confidence_threshold, float(confidence_threshold))
    final_readbacks = _option_snapshot(sensor)
    actual_confidence = final_readbacks["confidence_threshold"].get("value")
    if confidence_threshold is not None and (
        actual_confidence is None
        or not np.isclose(
            float(actual_confidence), float(confidence_threshold), atol=1e-6
        )
    ):
        raise RuntimeError(
            "L515 confidence readback mismatch: "
            f"requested={confidence_threshold}, actual={actual_confidence}"
        )
    return {
        "base_visual_preset": visual_preset,
        "confidence_override": confidence_threshold,
        "base_readbacks": base_readbacks,
        "final_readbacks": final_readbacks,
    }


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _region_nonzero_ratio(depth_raw: np.ndarray, region: Region | None) -> float:
    view = depth_raw if region is None else depth_raw[region.rows, region.columns]
    return float(np.count_nonzero(view) / view.size)


def _edge_flying_candidate_ratio(
    depth_raw: np.ndarray,
    *,
    depth_scale_m: float,
    edge_jump_m: float = 0.030,
    endpoint_band_m: float = 0.008,
) -> float:
    """Measure pixels selected by the production intermediate-depth rule."""
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = depth_raw > 0
    if not np.any(valid):
        return float("nan")
    local_min = minimum_filter(
        np.where(valid, depth_m, np.inf), size=3, mode="constant", cval=np.inf
    )
    local_max = maximum_filter(
        np.where(valid, depth_m, -np.inf), size=3, mode="constant", cval=-np.inf
    )
    local_valid_count = convolve(
        valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    endpoint_distance = np.minimum(depth_m - local_min, local_max - depth_m)
    candidate = (
        valid
        & (local_valid_count >= 3)
        & ((local_max - local_min) > edge_jump_m)
        & (endpoint_distance > endpoint_band_m)
    )
    return float(np.count_nonzero(candidate) / np.count_nonzero(valid))


def _point_vertices(points: Any, *, height: int, width: int) -> np.ndarray:
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    if vertices.shape != (height * width, 3):
        raise RuntimeError(f"unexpected SDK point array shape {vertices.shape}")
    return vertices.reshape(height, width, 3)


def _plane_rms_m(vertices: np.ndarray, region: Region) -> float:
    points = vertices[region.rows, region.columns].reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)]
    if points.shape[0] < 3:
        return float("nan")
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    distances = (points - center) @ normal
    return float(np.sqrt(np.mean(distances * distances)))


def _sdk_version() -> str:
    version = getattr(rs, "__version__", None)
    if version is not None:
        return str(version)
    get_api_version = getattr(rs, "get_api_version", None)
    return str(get_api_version()) if callable(get_api_version) else "unknown"


def _open_memmaps(
    output_dir: Path | None,
    *,
    frames: int,
    height: int,
    width: int,
    save_rgb: bool,
) -> tuple[np.memmap | None, np.memmap | None]:
    if output_dir is None:
        return None, None
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory must be absent or empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    depth = np.lib.format.open_memmap(
        output_dir / "native_depth_z16.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(frames, height, width),
    )
    rgb = None
    if save_rgb:
        rgb = np.lib.format.open_memmap(
            output_dir / "native_color_rgb8.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(frames, height, width, 3),
        )
    return depth, rgb


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    context = rs.context()
    serial = _select_serial(context, args.serial)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps
    )

    profile = None
    depth_memmap: np.memmap | None = None
    rgb_memmap: np.memmap | None = None
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        product_line = _device_info(device, rs.camera_info.product_line)
        name = _device_info(device, rs.camera_info.name)
        if product_line != "L500" and "L515" not in name.upper():
            raise RuntimeError(
                f"expected L515, got name={name!r}, product_line={product_line!r}"
            )

        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        t_color_from_depth = _extrinsics_matrix(
            depth_profile.get_extrinsics_to(color_profile)
        )
        depth_sensor = device.first_depth_sensor()
        depth_scale_m = float(depth_sensor.get_depth_scale())
        initial_option_readbacks = _option_snapshot(depth_sensor)
        applied_setting_variant = _apply_setting_variant(
            depth_sensor,
            visual_preset=args.visual_preset,
            confidence_threshold=args.confidence_threshold,
        )

        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(args.timeout_ms)

        depth_memmap, rgb_memmap = _open_memmaps(
            args.output_dir,
            frames=args.frames,
            height=args.height,
            width=args.width,
            save_rgb=args.save_rgb,
        )

        depth_frame_numbers: list[int] = []
        color_frame_numbers: list[int] = []
        depth_timestamps_s: list[float] = []
        color_timestamps_s: list[float] = []
        depth_timestamp_domains: list[int] = []
        color_timestamp_domains: list[int] = []
        skew_ms: list[float] = []
        payload_copy_ms: list[float] = []
        sdk_pointcloud_ms: list[float] = []
        depth_nonzero_ratio: list[float] = []
        foreground_retention: list[float] = []
        background_retention: list[float] = []
        flying_candidate_ratio: list[float] = []
        plane_rms_m: list[float] = []
        pointcloud = rs.pointcloud()

        for index in range(args.frames):
            frameset = pipeline.wait_for_frames(args.timeout_ms)
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                raise RuntimeError(f"frameset {index} is missing depth or color")

            copy_start_ns = time.monotonic_ns()
            # SDK frames are invalid after the next pipeline call. ``array``
            # with copy=True makes this timing an actual ownership-copy metric.
            depth_raw = np.array(
                depth_frame.get_data(), dtype=np.uint16, copy=True, order="C"
            )
            color_rgb = np.array(
                color_frame.get_data(), dtype=np.uint8, copy=True, order="C"
            )
            payload_copy_ms.append((time.monotonic_ns() - copy_start_ns) / 1e6)
            if (
                depth_raw.shape != (args.height, args.width)
                or depth_raw.dtype != np.uint16
            ):
                raise RuntimeError(
                    f"unexpected native depth shape/dtype {depth_raw.shape}/{depth_raw.dtype}"
                )
            if (
                color_rgb.shape != (args.height, args.width, 3)
                or color_rgb.dtype != np.uint8
            ):
                raise RuntimeError(
                    f"unexpected native color shape/dtype {color_rgb.shape}/{color_rgb.dtype}"
                )

            depth_number = int(depth_frame.get_frame_number())
            color_number = int(color_frame.get_frame_number())
            depth_timestamp_s = float(depth_frame.get_timestamp()) * 1e-3
            color_timestamp_s = float(color_frame.get_timestamp()) * 1e-3
            depth_timestamp_domain = int(depth_frame.get_frame_timestamp_domain())
            color_timestamp_domain = int(color_frame.get_frame_timestamp_domain())
            depth_frame_numbers.append(depth_number)
            color_frame_numbers.append(color_number)
            depth_timestamps_s.append(depth_timestamp_s)
            color_timestamps_s.append(color_timestamp_s)
            depth_timestamp_domains.append(depth_timestamp_domain)
            color_timestamp_domains.append(color_timestamp_domain)
            skew_ms.append(
                abs(depth_timestamp_s - color_timestamp_s) * 1e3
                if depth_timestamp_domain == color_timestamp_domain
                else float("nan")
            )

            depth_nonzero_ratio.append(_region_nonzero_ratio(depth_raw, None))
            if args.foreground_roi is not None:
                foreground_retention.append(
                    _region_nonzero_ratio(depth_raw, args.foreground_roi)
                )
            if args.background_roi is not None:
                background_retention.append(
                    _region_nonzero_ratio(depth_raw, args.background_roi)
                )
            flying_candidate_ratio.append(
                _edge_flying_candidate_ratio(depth_raw, depth_scale_m=depth_scale_m)
            )

            pointcloud_start_ns = time.monotonic_ns()
            points = pointcloud.calculate(depth_frame)
            sdk_pointcloud_ms.append((time.monotonic_ns() - pointcloud_start_ns) / 1e6)
            if args.plane_roi is not None:
                vertices = _point_vertices(points, height=args.height, width=args.width)
                plane_rms_m.append(_plane_rms_m(vertices, args.plane_roi))

            if depth_memmap is not None:
                depth_memmap[index] = depth_raw
            if rgb_memmap is not None:
                rgb_memmap[index] = color_rgb

        depth_frame_gaps = np.maximum(0, np.diff(depth_frame_numbers) - 1)
        color_frame_gaps = np.maximum(0, np.diff(color_frame_numbers) - 1)
        if args.output_dir is not None:
            np.savez(
                args.output_dir / "frame_timing.npz",
                depth_frame_number=np.asarray(depth_frame_numbers, dtype=np.uint64),
                color_frame_number=np.asarray(color_frame_numbers, dtype=np.uint64),
                depth_device_timestamp_s=np.asarray(
                    depth_timestamps_s, dtype=np.float64
                ),
                color_device_timestamp_s=np.asarray(
                    color_timestamps_s, dtype=np.float64
                ),
                depth_timestamp_domain=np.asarray(
                    depth_timestamp_domains, dtype=np.int32
                ),
                color_timestamp_domain=np.asarray(
                    color_timestamp_domains, dtype=np.int32
                ),
            )
        info = {
            "scene": args.scene,
            "serial": serial,
            "name": name,
            "firmware": _device_info(device, rs.camera_info.firmware_version),
            "sdk_version": _sdk_version(),
            "usb_mode": _device_info(
                device, getattr(rs.camera_info, "usb_type_descriptor", None)
            ),
            "product_line": product_line,
            "depth_scale_m": depth_scale_m,
            "depth_stream": {
                "profile": str(depth_profile),
                "fps": int(depth_profile.fps()),
                "format": str(depth_profile.format()),
                "intrinsics": _intrinsics_dict(depth_intrinsics),
                "timestamp_domain": str(
                    frameset.get_depth_frame().get_frame_timestamp_domain()
                ),
            },
            "color_stream": {
                "profile": str(color_profile),
                "fps": int(color_profile.fps()),
                "format": str(color_profile.format()),
                "intrinsics": _intrinsics_dict(color_intrinsics),
                "timestamp_domain": str(
                    frameset.get_color_frame().get_frame_timestamp_domain()
                ),
            },
            "T_color_from_depth": t_color_from_depth.tolist(),
            "l515_option_readbacks_initial": initial_option_readbacks,
            "l515_setting_variant": applied_setting_variant,
            "l515_option_readbacks_final": _option_snapshot(depth_sensor),
            "capture": {
                "frames": args.frames,
                "warmup_frames": args.warmup_frames,
                "depth_frame_number_first": depth_frame_numbers[0],
                "depth_frame_number_last": depth_frame_numbers[-1],
                "color_frame_number_first": color_frame_numbers[0],
                "color_frame_number_last": color_frame_numbers[-1],
                "depth_device_timestamp_s_first": depth_timestamps_s[0],
                "depth_device_timestamp_s_last": depth_timestamps_s[-1],
                "color_device_timestamp_s_first": color_timestamps_s[0],
                "color_device_timestamp_s_last": color_timestamps_s[-1],
                "depth_frame_gap_total": int(np.sum(depth_frame_gaps)),
                "color_frame_gap_total": int(np.sum(color_frame_gaps)),
                "matching_timestamp_domain_frames": int(
                    np.count_nonzero(
                        np.asarray(depth_timestamp_domains)
                        == np.asarray(color_timestamp_domains)
                    )
                ),
                "rgb_depth_abs_skew_ms": _percentiles(skew_ms),
                "payload_copy_ms": _percentiles(payload_copy_ms),
                "sdk_native_pointcloud_ms": _percentiles(sdk_pointcloud_ms),
            },
            "quality": {
                "depth_nonzero_ratio": _percentiles(depth_nonzero_ratio),
                "edge_flying_candidate_ratio": _percentiles(flying_candidate_ratio),
                "foreground_nonzero_retention": _percentiles(foreground_retention),
                "background_nonzero_retention": _percentiles(background_retention),
                "plane_rms_mm": _percentiles([value * 1e3 for value in plane_rms_m]),
            },
            "metric_definitions": {
                "edge_flying_candidate_ratio": (
                    "valid pixels with >=3 valid 3x3 samples, local span >30mm, "
                    "and distance from both local endpoints >8mm / valid pixels"
                ),
                "foreground_nonzero_retention": "nonzero native Z16 ratio in --foreground-roi",
                "background_nonzero_retention": "nonzero native Z16 ratio in --background-roi",
                "plane_rms_mm": "SVD plane residual over SDK-deprojected --plane-roi points",
            },
            "regions": {
                "plane_roi": (
                    None if args.plane_roi is None else args.plane_roi.to_list()
                ),
                "foreground_roi": (
                    None
                    if args.foreground_roi is None
                    else args.foreground_roi.to_list()
                ),
                "background_roi": (
                    None
                    if args.background_roi is None
                    else args.background_roi.to_list()
                ),
            },
        }
        return info
    finally:
        if depth_memmap is not None:
            depth_memmap.flush()
        if rgb_memmap is not None:
            rgb_memmap.flush()
        if profile is not None:
            pipeline.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = _capture(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L515 inspection failed: {exc}", file=sys.stderr)
        return 1

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(serialized)
    if args.output_dir is not None:
        report_path = args.output_dir / "report.json"
        report_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

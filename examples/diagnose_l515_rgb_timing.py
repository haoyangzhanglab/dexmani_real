#!/usr/bin/env python3
"""Diagnose L515 RGB stream cadence without modifying camera control.

This diagnostic connects to one RealSense camera and records stream timing,
frame numbers, per-frame metadata, and color option readback with minimal
host-side processing. It does not connect to or command the robot, open a GUI,
or write calibration. It observes camera control (``get_option``) but never
mutates it (no ``set_option``); the stream configuration matches the production
RGB-D path, but the option profile is left exactly as found.

It is the precise stream-cadence / RGB-FPS counterpart to ``inspect_l515.py``,
whose acquisition loop also computes geometry and depth quality. Use this
script when the question is "why is the RGB stream N Hz", not "what does the
depth look like".

Example::

    python examples/diagnose_l515_rgb_timing.py \
        --serial f1382055 \
        --mode rgbd \
        --label dark_rgbd_baseline_01 \
        --output-dir diagnostics/dark_rgbd_baseline_01
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
import pyrealsense2 as rs  # type: ignore[import-not-found]

# Frame-metadata enum names read per stream (see frame_metadata.md). Unsupported
# or absent members are recorded as NaN / a support flag rather than failing.
_COLOR_METADATA_READ = (
    "frame_counter",
    "frame_timestamp",
    "sensor_timestamp",
    "actual_exposure",
    "gain_level",
    "auto_exposure",
    "time_of_arrival",
    "backend_timestamp",
    "actual_fps",
    "exposure_priority",
)
_DEPTH_METADATA_READ = (
    "frame_counter",
    "frame_timestamp",
    "sensor_timestamp",
    "time_of_arrival",
    "backend_timestamp",
    "actual_fps",
)

# Color sensor options snapshotted at after-start / after-warmup / after-capture.
_COLOR_OPTION_NAMES = (
    "enable_auto_exposure",
    "auto_exposure_priority",
    "exposure",
    "gain",
    "power_line_frequency",
    "enable_auto_white_balance",
    "white_balance",
    "brightness",
    "contrast",
    "gamma",
    "saturation",
    "sharpness",
    "backlight_compensation",
)

# Evidence heuristic bands (section 28). These are thresholds, not conclusions.
_NOMINAL_33MS_PERIOD_MS = 1000.0 / 30.0
_NOMINAL_60MS_PERIOD_MS = 1000.0 / 16.68
_PERIOD_TOLERANCE_MS = 4.0
_EXPOSURE_60MS_US = 60_000.0
_EXPOSURE_TOLERANCE_US = 6_000.0


def _device_info(device: Any, key: Any) -> str:
    if key is None:
        return ""
    try:
        return str(device.get_info(key)) if device.supports(key) else ""
    except RuntimeError:
        return ""


def _sdk_version() -> str:
    version = getattr(rs, "__version__", None)
    if version is not None:
        return str(version)
    get_api_version = getattr(rs, "get_api_version", None)
    return str(get_api_version()) if callable(get_api_version) else "unknown"


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
            "connect exactly one RealSense or pass --serial; connected={serials}"
        )
    return serials[0]


def _metadata_value(frame: Any, name: str) -> float:
    value = getattr(rs.frame_metadata_value, name, None)
    if value is None or not frame.supports_frame_metadata(value):
        return float("nan")
    return float(frame.get_frame_metadata(value))


def _metadata_supported(frame: Any, name: str) -> bool:
    value = getattr(rs.frame_metadata_value, name, None)
    return value is not None and frame.supports_frame_metadata(value)


def _find_color_sensor(device: Any) -> Any:
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                if profile.stream_type() == rs.stream.color:
                    return sensor
            except RuntimeError:
                continue
    raise RuntimeError("no color sensor with a color stream profile found")


def _color_option_snapshot(sensor: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _COLOR_OPTION_NAMES:
        option = getattr(rs.option, name, None)
        if option is None:
            result[name] = {"exposed_by_wrapper": False}
            continue
        try:
            supported = bool(sensor.supports(option))
        except RuntimeError:
            supported = False
        if not supported:
            result[name] = {"exposed_by_wrapper": True, "supported_by_sensor": False}
            continue
        item: dict[str, Any] = {
            "exposed_by_wrapper": True,
            "supported_by_sensor": True,
            "value": float(sensor.get_option(option)),
        }
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
        result[name] = item
    return result


def _option_value(snapshot: dict[str, Any], name: str) -> float | None:
    item = snapshot.get(name)
    if isinstance(item, dict) and item.get("supported_by_sensor"):
        value = item.get("value")
        return float(value) if value is not None else None
    return None


def _video_profile_dict(video: Any) -> dict[str, Any]:
    return {
        "width": int(video.width()),
        "height": int(video.height()),
        "fps": int(video.fps()),
        "format": str(video.format()),
        "stream_index": int(video.stream_index()),
    }


def _frameset_frame_summary(frameset: Any) -> str:
    parts: list[str] = []
    try:
        size = frameset.size()
    except RuntimeError:
        size = 0
    for index in range(size):
        try:
            profile = frameset[index].get_profile()
            parts.append(f"{profile.stream_type()}/{profile.format()}")
        except RuntimeError:
            continue
    return ", ".join(parts) if parts else "<no frames>"


def _extract_color_frame(queued_frame: Any) -> Any:
    """Return a color frame from either a frameset or a single video frame.

    pyrealsense2 returns an empty (falsy) frame object, not ``None``, when a
    frameset lacks a stream, so truthiness is the correct absence test.
    """
    frameset = queued_frame.as_frameset()
    if frameset:
        color = frameset.get_color_frame()
        if color:
            return color
        raise RuntimeError(
            "color queue returned a frameset with no color frame: "
            + _frameset_frame_summary(frameset)
        )
    profile = queued_frame.get_profile()
    if profile.stream_type() == rs.stream.color:
        return queued_frame
    raise RuntimeError(
        "queued frame is neither a color frame nor a frameset with color "
        f"(stream_type={profile.stream_type()}, format={profile.format()})"
    )


def _extract_depth_color(queued_frame: Any) -> tuple[Any, Any]:
    """Require a composite frameset containing both depth and color frames."""
    frameset = queued_frame.as_frameset()
    if not frameset:
        profile = queued_frame.get_profile()
        raise RuntimeError(
            "RGB-D queue returned a non-frameset frame "
            f"(stream_type={profile.stream_type()}, format={profile.format()})"
        )
    depth = frameset.get_depth_frame()
    color = frameset.get_color_frame()
    if not depth or not color:
        raise RuntimeError(
            "RGB-D frameset missing depth or color frame: "
            + _frameset_frame_summary(frameset)
        )
    return depth, color


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return None if not np.isfinite(value) else float(value)


def _dedupe_consecutive(
    frame_numbers: np.ndarray, timestamps_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop consecutive repeated frame numbers (repeated observations)."""
    numbers = np.asarray(frame_numbers, dtype=np.int64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if numbers.size < 2:
        return numbers, timestamps
    keep = np.empty(numbers.size, dtype=bool)
    keep[0] = True
    keep[1:] = numbers[1:] != numbers[:-1]
    return numbers[keep], timestamps[keep]


def _unique_rate_hz(frame_numbers: np.ndarray, timestamps_s: np.ndarray) -> float:
    numbers, timestamps = _dedupe_consecutive(frame_numbers, timestamps_s)
    if numbers.size < 2:
        return float("nan")
    span_s = timestamps[-1] - timestamps[0]
    if not np.isfinite(span_s) or span_s <= 0.0:
        return float("nan")
    return float(numbers.size - 1) / span_s


def _frame_gap_stats(frame_numbers: np.ndarray) -> tuple[int, int, int]:
    """Return (gap_event_count, missing_total, reset_or_rollback_count)."""
    numbers, _ = _dedupe_consecutive(
        frame_numbers, np.zeros_like(frame_numbers, dtype=np.float64)
    )
    if numbers.size < 2:
        return 0, 0, 0
    deltas = np.diff(numbers)
    gap_events = int(np.count_nonzero(deltas > 1))
    missing_total = int(np.sum(deltas[deltas > 1] - 1))
    reset_rollback = int(np.count_nonzero(deltas <= 0))
    return gap_events, missing_total, reset_rollback


def _normalized_period_ms(
    frame_numbers: np.ndarray, timestamps_s: np.ndarray
) -> np.ndarray:
    """Per distinct frame: device-time delta divided by frame-number delta."""
    numbers, timestamps = _dedupe_consecutive(frame_numbers, timestamps_s)
    if numbers.size < 2:
        return np.empty(0, dtype=np.float64)
    dt_s = np.diff(timestamps)
    dn = np.diff(numbers)
    with np.errstate(divide="ignore", invalid="ignore"):
        period_s = np.where(dn > 0, dt_s / dn, np.nan)
    return period_s * 1e3


def _unique_interval_ms(
    frame_numbers: np.ndarray, timestamps_s: np.ndarray
) -> np.ndarray:
    _, timestamps = _dedupe_consecutive(frame_numbers, timestamps_s)
    if timestamps.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(timestamps) * 1e3


def _bgr8_luma(bgr: np.ndarray) -> np.ndarray:
    return (
        0.114 * bgr[..., 0].astype(np.float32)
        + 0.587 * bgr[..., 1].astype(np.float32)
        + 0.299 * bgr[..., 2].astype(np.float32)
    )


def _luma_stats(luma: np.ndarray) -> dict[str, float]:
    flat = luma.ravel().astype(np.float32)
    return {
        "mean": float(np.mean(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "black_ratio": float(np.count_nonzero(flat < 16.0) / flat.size),
        "highlight_clip_ratio": float(np.count_nonzero(flat > 245.0) / flat.size),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose L515 RGB/D stream cadence without modifying camera control."
    )
    parser.add_argument("--serial", default=None)
    parser.add_argument("--mode", choices=("rgbd", "color"), default="rgbd")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=None,
        help="Override the per-mode default (2 for rgbd, 1 for color).",
    )
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--sample-luma-every", type=int, default=30)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if args.warmup_seconds < 0 or args.duration_seconds <= 0 or args.timeout_ms <= 0:
        parser.error("warmup must be non-negative; duration and timeout must be positive")
    if args.sample_luma_every <= 0:
        parser.error("--sample-luma-every must be positive")
    if args.queue_capacity is not None and args.queue_capacity <= 0:
        parser.error("--queue-capacity must be positive when supplied")
    return args


def _capture(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    queue_capacity = (
        args.queue_capacity if args.queue_capacity is not None else (2 if args.mode == "rgbd" else 1)
    )

    context = rs.context()
    serial = _select_serial(context, args.serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    if args.mode == "rgbd":
        config.enable_stream(
            rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
        )
    queue = rs.frame_queue(queue_capacity)

    profile = None
    # Per-queued-frame scalar arrays (index i = one queued frame).
    host_wait_return_ns: list[int] = []
    depth_frame_numbers: list[int] = []
    depth_timestamps_s: list[float] = []
    depth_timestamp_domains: list[int] = []
    color_frame_numbers: list[int] = []
    color_timestamps_s: list[float] = []
    color_timestamp_domains: list[int] = []
    # Per-frame color metadata (nan when unsupported).
    color_frame_counter: list[float] = []
    color_frame_timestamp_us: list[float] = []
    color_sensor_timestamp_us: list[float] = []
    color_actual_exposure_us: list[float] = []
    color_gain_level: list[float] = []
    color_auto_exposure: list[float] = []
    color_exposure_priority: list[float] = []
    color_actual_fps_hz: list[float] = []
    color_backend_timestamp_us: list[float] = []
    color_time_of_arrival_us: list[float] = []
    # Per-frame depth metadata.
    depth_frame_counter: list[float] = []
    depth_frame_timestamp_us: list[float] = []
    depth_sensor_timestamp_us: list[float] = []
    depth_time_of_arrival_us: list[float] = []
    depth_backend_timestamp_us: list[float] = []
    depth_actual_fps_hz: list[float] = []

    color_metadata_supported: dict[str, bool] = {}
    depth_metadata_supported: dict[str, bool] = {}
    luma_samples: list[np.ndarray] = []
    warmup_frames = 0

    try:
        profile = pipeline.start(config, queue)
        device = profile.get_device()
        color_sensor = _find_color_sensor(device)
        options_after_start = _color_option_snapshot(color_sensor)

        warmup_deadline_ns = time.monotonic_ns() + int(args.warmup_seconds * 1e9)
        while time.monotonic_ns() < warmup_deadline_ns:
            queue.wait_for_frame(args.timeout_ms)
            warmup_frames += 1
        options_after_warmup = _color_option_snapshot(color_sensor)

        capture_deadline_ns = time.monotonic_ns() + int(args.duration_seconds * 1e9)
        prev_color_number: int | None = None
        color_unique_observed = 0

        while time.monotonic_ns() < capture_deadline_ns:
            queued_frame = queue.wait_for_frame(args.timeout_ms)
            wait_return_ns = time.monotonic_ns()
            if args.mode == "rgbd":
                depth_frame, color_frame = _extract_depth_color(queued_frame)
            else:
                depth_frame = None
                color_frame = _extract_color_frame(queued_frame)

            host_wait_return_ns.append(wait_return_ns)
            color_number = int(color_frame.get_frame_number())
            color_frame_numbers.append(color_number)
            color_timestamps_s.append(float(color_frame.get_timestamp()) * 1e-3)
            color_timestamp_domains.append(
                int(color_frame.get_frame_timestamp_domain())
            )

            if not color_metadata_supported:
                color_metadata_supported = {
                    name: _metadata_supported(color_frame, name)
                    for name in _COLOR_METADATA_READ
                }
            color_frame_counter.append(_metadata_value(color_frame, "frame_counter"))
            color_frame_timestamp_us.append(
                _metadata_value(color_frame, "frame_timestamp")
            )
            color_sensor_timestamp_us.append(
                _metadata_value(color_frame, "sensor_timestamp")
            )
            color_actual_exposure_us.append(
                _metadata_value(color_frame, "actual_exposure")
            )
            color_gain_level.append(_metadata_value(color_frame, "gain_level"))
            color_auto_exposure.append(_metadata_value(color_frame, "auto_exposure"))
            color_exposure_priority.append(
                _metadata_value(color_frame, "exposure_priority")
            )
            color_actual_fps_hz.append(
                _metadata_value(color_frame, "actual_fps") / 1000.0
            )
            color_backend_timestamp_us.append(
                _metadata_value(color_frame, "backend_timestamp")
            )
            color_time_of_arrival_us.append(
                _metadata_value(color_frame, "time_of_arrival")
            )

            if depth_frame is not None:
                depth_frame_numbers.append(int(depth_frame.get_frame_number()))
                depth_timestamps_s.append(float(depth_frame.get_timestamp()) * 1e-3)
                depth_timestamp_domains.append(
                    int(depth_frame.get_frame_timestamp_domain())
                )
                if not depth_metadata_supported:
                    depth_metadata_supported = {
                        name: _metadata_supported(depth_frame, name)
                        for name in _DEPTH_METADATA_READ
                    }
                depth_frame_counter.append(
                    _metadata_value(depth_frame, "frame_counter")
                )
                depth_frame_timestamp_us.append(
                    _metadata_value(depth_frame, "frame_timestamp")
                )
                depth_sensor_timestamp_us.append(
                    _metadata_value(depth_frame, "sensor_timestamp")
                )
                depth_time_of_arrival_us.append(
                    _metadata_value(depth_frame, "time_of_arrival")
                )
                depth_backend_timestamp_us.append(
                    _metadata_value(depth_frame, "backend_timestamp")
                )
                depth_actual_fps_hz.append(
                    _metadata_value(depth_frame, "actual_fps") / 1000.0
                )

            is_new_color = prev_color_number is None or color_number != prev_color_number
            if is_new_color:
                color_unique_observed += 1
                if color_unique_observed % args.sample_luma_every == 0:
                    bgr = np.asanyarray(color_frame.get_data())
                    luma_samples.append(_bgr8_luma(bgr).astype(np.float32))
            prev_color_number = color_number

        options_after_capture = _color_option_snapshot(color_sensor)

        active_profiles: dict[str, Any] = {}
        for name, stream in (("color", rs.stream.color), ("depth", rs.stream.depth)):
            if name == "depth" and args.mode != "rgbd":
                continue
            active_profiles[name] = _video_profile_dict(
                profile.get_stream(stream).as_video_stream_profile()
            )
        options: dict[str, Any] = {
            "device": {
                "name": _device_info(device, rs.camera_info.name),
                "serial": serial,
                "firmware": _device_info(device, rs.camera_info.firmware_version),
                "product_line": _device_info(device, rs.camera_info.product_line),
                "usb_type_descriptor": _device_info(
                    device, getattr(rs.camera_info, "usb_type_descriptor", None)
                ),
                "librealsense_version": _sdk_version(),
            },
            "active_profiles": active_profiles,
            "color_sensor": {
                "after_start": options_after_start,
                "after_warmup": options_after_warmup,
                "after_capture": options_after_capture,
            },
        }
    finally:
        if profile is not None:
            pipeline.stop()

    depth_numbers = np.asarray(depth_frame_numbers, dtype=np.int64)
    depth_ts = np.asarray(depth_timestamps_s, dtype=np.float64)
    depth_domains = np.asarray(depth_timestamp_domains, dtype=np.int32)
    color_numbers = np.asarray(color_frame_numbers, dtype=np.int64)
    color_ts = np.asarray(color_timestamps_s, dtype=np.float64)
    color_domains = np.asarray(color_timestamp_domains, dtype=np.int32)

    n_framesets = len(host_wait_return_ns)
    if n_framesets >= 2:
        host_span_s = (host_wait_return_ns[-1] - host_wait_return_ns[0]) / 1e9
        host_output_rate_hz = n_framesets / host_span_s if host_span_s > 0.0 else float("nan")
    else:
        host_span_s = float("nan")
        host_output_rate_hz = float("nan")

    color_is_repeated = np.zeros(n_framesets, dtype=bool)
    if args.mode == "rgbd" and n_framesets >= 2:
        color_is_repeated[1:] = color_numbers[1:] == color_numbers[:-1]
    color_repeat_count = int(np.count_nonzero(color_is_repeated))
    color_repeat_ratio = color_repeat_count / n_framesets if n_framesets else float("nan")

    color_unique_rate_hz = _unique_rate_hz(color_numbers, color_ts)
    depth_unique_rate_hz = (
        _unique_rate_hz(depth_numbers, depth_ts) if args.mode == "rgbd" else None
    )

    color_gap_events, color_missing_total, color_resets = _frame_gap_stats(color_numbers)
    if args.mode == "rgbd":
        depth_gap_events, depth_missing_total, depth_resets = _frame_gap_stats(
            depth_numbers
        )
    else:
        depth_gap_events = depth_missing_total = depth_resets = None
    reset_rollback_count = color_resets + (depth_resets if depth_resets is not None else 0)

    color_unique_interval_ms = _unique_interval_ms(color_numbers, color_ts)
    color_normalized_period_ms = _normalized_period_ms(color_numbers, color_ts)

    rgbd_skew_s = np.full(n_framesets, np.nan, dtype=np.float64)
    cross_stream_device_timestamp_domains_match: bool | None = None
    if args.mode == "rgbd" and n_framesets:
        same_domain = depth_domains == color_domains
        rgbd_skew_s[same_domain] = depth_ts[same_domain] - color_ts[same_domain]
        cross_stream_device_timestamp_domains_match = bool(np.all(same_domain))
    rgbd_skew_all_ms = rgbd_skew_s * 1e3
    rgbd_skew_new_color_ms = rgbd_skew_s[~color_is_repeated] * 1e3

    luminance: dict[str, Any]
    if luma_samples:
        all_luma = np.concatenate([luma.ravel() for luma in luma_samples])
        luminance = _luma_stats(all_luma)
        luminance["sample_count"] = len(luma_samples)
    else:
        luminance = {
            "sample_count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "black_ratio": None,
            "highlight_clip_ratio": None,
        }

    color_period_median = (
        float(np.nanmedian(color_normalized_period_ms))
        if color_normalized_period_ms.size
        else float("nan")
    )
    actual_exposure_median_us = (
        float(np.nanmedian(np.asarray(color_actual_exposure_us, dtype=np.float64)))
        if color_actual_exposure_us
        else float("nan")
    )
    auto_exposure_enabled = _option_value(options_after_capture, "enable_auto_exposure")
    auto_exposure_priority = _option_value(
        options_after_capture, "auto_exposure_priority"
    )
    if np.isfinite(actual_exposure_median_us):
        actual_exposure_near_60ms: bool | None = bool(
            abs(actual_exposure_median_us - _EXPOSURE_60MS_US)
            <= _EXPOSURE_TOLERANCE_US
        )
    else:
        actual_exposure_near_60ms = None

    evidence: dict[str, Any] = {
        "color_frame_numbers_contiguous": bool(
            color_gap_events == 0 and np.isfinite(color_period_median)
        ),
        "color_normalized_period_near_33ms": bool(
            np.isfinite(color_period_median)
            and abs(color_period_median - _NOMINAL_33MS_PERIOD_MS)
            <= _PERIOD_TOLERANCE_MS
        ),
        "color_normalized_period_near_60ms": bool(
            np.isfinite(color_period_median)
            and abs(color_period_median - _NOMINAL_60MS_PERIOD_MS)
            <= _PERIOD_TOLERANCE_MS
        ),
        "auto_exposure_enabled": auto_exposure_enabled,
        "auto_exposure_priority": auto_exposure_priority,
        "actual_exposure_near_60ms": actual_exposure_near_60ms,
    }

    depth_numbers_unique, _ = _dedupe_consecutive(depth_numbers, depth_ts)
    color_numbers_unique, _ = _dedupe_consecutive(color_numbers, color_ts)
    depth_unique_count = (
        int(depth_numbers_unique.size) if args.mode == "rgbd" else None
    )
    color_unique_count = int(color_numbers_unique.size)
    frame_arrays: dict[str, np.ndarray] = {
        "host_wait_return_monotonic_ns": np.asarray(host_wait_return_ns, dtype=np.int64),
        "depth_frame_number": depth_numbers,
        "depth_device_timestamp_s": depth_ts,
        "depth_timestamp_domain": depth_domains,
        "color_frame_number": color_numbers,
        "color_device_timestamp_s": color_ts,
        "color_timestamp_domain": color_domains,
        "color_is_repeated": color_is_repeated,
        "depth_frame_number_delta": (
            np.diff(depth_numbers_unique).astype(np.int64)
            if depth_numbers_unique.size >= 2
            else np.empty(0, dtype=np.int64)
        ),
        "color_frame_number_delta": (
            np.diff(color_numbers_unique).astype(np.int64)
            if color_numbers_unique.size >= 2
            else np.empty(0, dtype=np.int64)
        ),
        "rgbd_device_skew_s": rgbd_skew_s,
        "depth_frame_counter": np.asarray(depth_frame_counter, dtype=np.float64),
        "depth_frame_timestamp_us": np.asarray(
            depth_frame_timestamp_us, dtype=np.float64
        ),
        "depth_sensor_timestamp_us": np.asarray(
            depth_sensor_timestamp_us, dtype=np.float64
        ),
        "depth_time_of_arrival_us": np.asarray(
            depth_time_of_arrival_us, dtype=np.float64
        ),
        "depth_backend_timestamp_us": np.asarray(
            depth_backend_timestamp_us, dtype=np.float64
        ),
        "depth_actual_fps_hz": np.asarray(depth_actual_fps_hz, dtype=np.float64),
        "color_frame_counter": np.asarray(color_frame_counter, dtype=np.float64),
        "color_frame_timestamp_us": np.asarray(
            color_frame_timestamp_us, dtype=np.float64
        ),
        "color_sensor_timestamp_us": np.asarray(
            color_sensor_timestamp_us, dtype=np.float64
        ),
        "color_actual_exposure_us": np.asarray(
            color_actual_exposure_us, dtype=np.float64
        ),
        "color_gain_level": np.asarray(color_gain_level, dtype=np.float64),
        "color_auto_exposure": np.asarray(color_auto_exposure, dtype=np.float64),
        "color_exposure_priority": np.asarray(
            color_exposure_priority, dtype=np.float64
        ),
        "color_actual_fps_hz": np.asarray(color_actual_fps_hz, dtype=np.float64),
        "color_backend_timestamp_us": np.asarray(
            color_backend_timestamp_us, dtype=np.float64
        ),
        "color_time_of_arrival_us": np.asarray(
            color_time_of_arrival_us, dtype=np.float64
        ),
    }

    report: dict[str, Any] = {
        "mode": args.mode,
        "label": args.label,
        "serial": serial,
        "width": args.width,
        "height": args.height,
        "requested_fps": args.fps,
        "queue_capacity": queue_capacity,
        "warmup_seconds": args.warmup_seconds,
        "duration_seconds": args.duration_seconds,
        "timeout_ms": args.timeout_ms,
        "sample_luma_every": args.sample_luma_every,
        "capture_duration_seconds": _finite_or_none(host_span_s),
        "stream_path": "production_profile_matched",
        "camera_options": "observed_not_modified",
        "frameset_count": n_framesets,
        "warmup_frames_observed": warmup_frames,
        "host_output_rate_hz": _finite_or_none(host_output_rate_hz),
        "depth_unique_rate_hz": _finite_or_none(depth_unique_rate_hz),
        "color_unique_rate_hz": _finite_or_none(color_unique_rate_hz),
        "depth_unique_count": depth_unique_count,
        "color_unique_count": color_unique_count,
        "color_repeat_count": color_repeat_count,
        "color_repeat_ratio": _finite_or_none(color_repeat_ratio),
        "depth_frame_gap_event_count": depth_gap_events,
        "depth_missing_frame_number_total": depth_missing_total,
        "color_frame_gap_event_count": color_gap_events,
        "color_missing_frame_number_total": color_missing_total,
        "frame_number_reset_or_rollback_count": reset_rollback_count,
        "color_unique_interval_ms": _percentiles(color_unique_interval_ms),
        "color_normalized_period_ms": _percentiles(color_normalized_period_ms),
        "rgbd_skew_all_ms": _percentiles(rgbd_skew_all_ms),
        "rgbd_skew_new_color_ms": _percentiles(rgbd_skew_new_color_ms),
        "cross_stream_device_timestamp_domains_match": (
            cross_stream_device_timestamp_domains_match
        ),
        "color_actual_exposure_us": _percentiles(color_actual_exposure_us),
        "color_gain_level": _percentiles(color_gain_level),
        "color_actual_fps_hz": _percentiles(color_actual_fps_hz),
        "luminance": luminance,
        "evidence": evidence,
        "metadata_support": {
            "color": color_metadata_supported,
            "depth": depth_metadata_supported,
        },
    }
    return report, frame_arrays, options


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    try:
        report, frame_arrays, options = _capture(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L515 RGB timing diagnostic failed: {exc}", file=sys.stderr)
        return 1

    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        print(f"output directory must be absent or empty: {output_dir}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output_dir / "report.json").write_text(report_json, encoding="utf-8")
    options_json = (
        json.dumps(options, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "options.json").write_text(options_json, encoding="utf-8")
    np.savez(output_dir / "frame_timing.npz", **frame_arrays)

    print(report_json, end="")
    print(
        f"Wrote {output_dir / 'report.json'}, {output_dir / 'options.json'}, "
        f"{output_dir / 'frame_timing.npz'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

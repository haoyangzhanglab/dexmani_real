"""Bounded, read-only RealSense diagnostics.

Hardware and GUI dependencies are imported only after ``main()`` selects a
mode, so importing this module is safe in offline tooling and pytest.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_DEPTH_DISPLAY_MIN_M = 0.3
_DEPTH_DISPLAY_MAX_M = 1.5


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {text!r}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only RealSense diagnostics")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("list", "lifecycle", "stream"),
        default="stream",
        help="list devices, validate one connect/read/disconnect cycle, or stream for a bounded duration",
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--serial", default=None, help="Camera serial override")
    parser.add_argument("--duration-s", type=_positive_float, default=10.0, help="Stream duration (default: 10)")
    parser.add_argument("--timeout-ms", type=_positive_int, default=5000, help="Per-frame read timeout")
    parser.add_argument("--display", action="store_true", help="Show a bounded RGB-D window during stream mode")
    return parser


def _make_camera(args: argparse.Namespace) -> Any:
    from dexmani_real.config.runtime import resolve_runtime_config
    from dexmani_real.sensor.realsense import RealSense, RealSenseConfig

    runtime = resolve_runtime_config(
        yaml_path=args.config,
        cli_overrides={"camera.serial": args.serial},
    )
    camera_cfg = runtime.camera
    config = RealSenseConfig(
        serial=camera_cfg.serial,
        depth_resolution=(int(camera_cfg.width), int(camera_cfg.height)),
        color_resolution=(int(camera_cfg.width), int(camera_cfg.height)),
        fps=int(camera_cfg.fps),
        align_mode=cast(Literal["depth_to_color", "color_to_depth", "none"], str(camera_cfg.align_mode)),
        warmup_frames=int(camera_cfg.warmup_frames),
    )
    return RealSense(config)


def _list_devices() -> int:
    from dexmani_real.sensor.realsense import RealSense

    cameras = RealSense.list_cameras()
    if not cameras:
        print("No RealSense camera detected.")
        return 1
    for index, camera in enumerate(cameras):
        print(
            f"[{index}] {camera.get('name', 'unknown')}  "
            f"serial={camera.get('serial', '')}  firmware={camera.get('firmware', '')}"
        )
    return 0


def _validate_frame(frame: Any) -> tuple[int, int, float]:
    depth = np.asarray(frame.depth)
    intrinsics = np.asarray(frame.K)
    if depth.ndim != 2 or depth.size == 0:
        raise RuntimeError(f"invalid depth shape {depth.shape}")
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise RuntimeError("camera intrinsics are invalid")
    if not math.isfinite(float(frame.depth_scale)) or float(frame.depth_scale) <= 0.0:
        raise RuntimeError("camera depth scale is invalid")
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise RuntimeError("depth frame contains no valid measurements")
    valid_ratio = float(valid_count / depth.size)
    return int(depth.shape[1]), int(depth.shape[0]), valid_ratio


def _run_lifecycle(args: argparse.Namespace) -> int:
    camera = _make_camera(args)
    try:
        if not camera.connect():
            raise RuntimeError("RealSense connect failed")
        frame = camera.read(timeout_ms=args.timeout_ms)
        width, height, valid_ratio = _validate_frame(frame)
        if bool(frame.duplicate):
            raise RuntimeError("RealSense lifecycle read returned a duplicate frame")
        info = camera.get_device_info()
        print(
            f"OK name={info.get('name', '')} serial={info.get('serial', '')} "
            f"frame={width}x{height} valid_depth={valid_ratio:.3f}"
        )
        return 0
    finally:
        camera.disconnect()


def _run_stream(args: argparse.Namespace) -> int:
    camera = _make_camera(args)
    cv2: Any | None = None
    make_depth_vis: Any | None = None
    if args.display:
        import cv2 as _cv2

        from dexmani_real.utils.pointcloud_utils import make_depth_vis as _make_depth_vis

        cv2 = _cv2
        make_depth_vis = _make_depth_vis

    frame_count = 0
    fresh_frame_count = 0
    duplicate_count = 0
    gap_count = 0
    read_times_ms: list[float] = []
    valid_ratios: list[float] = []
    started_s = time.monotonic()
    try:
        if not camera.connect():
            raise RuntimeError("RealSense connect failed")
        started_s = time.monotonic()
        deadline_s = started_s + args.duration_s
        print(f"Streaming for at most {args.duration_s:.1f}s; press Q/Esc to stop early.")
        while time.monotonic() < deadline_s:
            read_started_s = time.monotonic()
            frame = camera.read(timeout_ms=args.timeout_ms)
            read_times_ms.append((time.monotonic() - read_started_s) * 1000.0)
            width, height, valid_ratio = _validate_frame(frame)
            valid_ratios.append(valid_ratio)
            frame_count += 1
            duplicate = bool(frame.duplicate)
            duplicate_count += int(duplicate)
            fresh_frame_count += int(not duplicate)
            gap_count += max(0, int(frame.frame_gap))

            if cv2 is not None and make_depth_vis is not None:
                depth_panel = make_depth_vis(frame.depth, _DEPTH_DISPLAY_MIN_M, _DEPTH_DISPLAY_MAX_M)
                if frame.rgb is None:
                    panel = depth_panel
                else:
                    color_bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
                    if depth_panel.shape[:2] != color_bgr.shape[:2]:
                        depth_panel = cv2.resize(depth_panel, (color_bgr.shape[1], color_bgr.shape[0]))
                    panel = np.concatenate((color_bgr, depth_panel), axis=1)
                cv2.imshow("RealSense diagnostic", panel)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if frame_count == 1 or frame_count % max(1, int(camera.config.fps)) == 0:
                elapsed_s = max(time.monotonic() - started_s, 1e-9)
                print(
                    f"frames={frame_count} size={width}x{height} fps={frame_count / elapsed_s:.1f} "
                    f"read={np.mean(read_times_ms):.1f}ms valid={np.mean(valid_ratios):.3f} "
                    f"fresh={fresh_frame_count} duplicates={duplicate_count} gaps={gap_count}",
                    flush=True,
                )
    finally:
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                logger.warning("OpenCV window cleanup failed", exc_info=True)
        camera.disconnect()

    elapsed_s = max(time.monotonic() - started_s, 1e-9)
    print(
        f"Done: frames={frame_count} elapsed={elapsed_s:.2f}s fps={frame_count / elapsed_s:.2f} "
        f"fresh={fresh_frame_count} duplicates={duplicate_count} gaps={gap_count}"
    )
    if fresh_frame_count == 0:
        print("No fresh RealSense frame was observed.")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.serial is not None and not args.serial.strip():
        raise SystemExit("--serial must be non-empty")
    try:
        if args.mode == "list":
            return _list_devices()
        if args.mode == "lifecycle":
            return _run_lifecycle(args)
        return _run_stream(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception:
        logger.error("RealSense diagnostic failed", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

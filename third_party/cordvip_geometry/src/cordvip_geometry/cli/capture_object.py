#!/usr/bin/env python3
"""RGB-D 冻结与保存窗口；默认连接 RealSense，--rgbd-dir 使用离线源。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import shlex
import sys
import time

from ..io import new_output, write_json


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument("--rgbd-dir", type=Path, help="只读回放六位编号的 RGB-D NPZ，不连接相机")
    parser.add_argument("--replay-fps", type=float, default=10.0)
    parser.add_argument("--serial", help="RealSense serial；仅用于实时相机")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的新输出目录")
    args = parser.parse_args(argv)
    if args.rgbd_dir is not None and args.serial is not None:
        parser.error("--serial 与 --rgbd-dir 不能同时使用")
    if args.rgbd_dir is not None and (not math.isfinite(args.replay_fps) or args.replay_fps <= 0):
        parser.error("--replay-fps 必须为有限正数")
    if args.rgbd_dir is None and min(args.width, args.height, args.fps) < 1:
        parser.error("width、height、fps 必须为正整数")
    return args


def save_frame(output, frame, *, frozen_ns, confirmed_ns, mode):
    from PIL import Image
    from ..rgbd import save_frame

    Image.fromarray(frame.rgb).save(output / "rgb.png")
    save_frame(output / "frame.npz", frame)
    metadata = {
        "source_mode": mode,
        "source_id": frame.source_id,
        "timestamp_ns": frame.timestamp_ns,
        "clock": frame.clock,
        "row_index": frame.row_index,
        "frozen_host_monotonic_ns": frozen_ns,
        "confirmed_host_monotonic_ns": confirmed_ns,
        "rgb_file": "rgb.png",
        "rgbd_file": "frame.npz",
        "rgb_shape": list(frame.rgb.shape),
        "K": frame.K.tolist(),
        "distortion_model": frame.distortion_model,
        "distortion_coeffs": list(frame.distortion_coeffs),
        "depth_units": "m",
        "depth_storage": "original_source_frame_aligned_float32; zero_is_invalid",
        "pixel_grid": "source_color_camera; no_undistortion_or_resize",
        "selection": "space_freeze_then_enter_confirm",
    }
    write_json(output / "capture.json", metadata)
    return metadata


def panel(frame, state, message, cv2, np):
    if frame is None:
        rgb = np.full((480, 640, 3), 25, dtype=np.uint8)
        depth_color = rgb.copy()
    else:
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        # This fixed display scale affects only the GUI, never saved depth.
        depth_u8 = np.clip(frame.depth_m * (255.0 / 2.0), 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color[frame.depth_m == 0] = 0
    height = 480
    width = max(1, round(rgb.shape[1] * height / rgb.shape[0]))
    rgb = cv2.resize(rgb, (width, height))
    depth_color = cv2.resize(depth_color, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((height + 112, width * 2, 3), dtype=np.uint8)
    canvas[64 : 64 + height, :width] = rgb
    canvas[64 : 64 + height, width:] = depth_color
    lines = [
        f"{state} | space: freeze/resume | Enter: save frozen frame | q/Esc: exit",
        message,
        f"RGB | aligned depth display 0..2 m | source: {frame.source_id if frame else 'waiting'}",
    ]
    for index, line in enumerate(lines):
        y = (24, 50, height + 94)[index]
        cv2.putText(
            canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA
        )
    return canvas


def run(args):
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("拍照需要图形会话；--help 不需要图形会话或相机")
    import cv2
    import numpy as np
    from ..live_workers import LatestFrameReader
    from ..rgbd import iter_rgbd_directory
    from ..integrations.dexmani import iter_live_realsense

    # Device imports remain inside iter_live_realsense, consumed only by its owner.
    live = args.rgbd_dir is None
    source = (
        (
            lambda: iter_live_realsense(
                serial=args.serial, width=args.width, height=args.height, fps=args.fps
            )
        )
        if live
        else (lambda: iter_rgbd_directory(args.rgbd_dir))
    )
    reader = LatestFrameReader(source, replay_interval_s=0 if live else 1.0 / args.replay_fps)
    output = new_output(args.output_dir, workspace=args.workspace, protected=(args.rgbd_dir,))
    write_json(
        output / "run.json",
        {
            "source_mode": "live_realsense" if live else "offline_replay",
            "rgbd_directory": str(args.rgbd_dir.expanduser().resolve()) if args.rgbd_dir else None,
            "requested_camera": (
                {"serial": args.serial, "width": args.width, "height": args.height, "fps": args.fps}
                if live
                else None
            ),
            "replay_fps": args.replay_fps if not live else None,
            "started_host_monotonic_ns": time.monotonic_ns(),
        },
    )
    title = "Capture one object | RGB and aligned depth"
    latest = frozen = None
    frozen_ns = None
    saved = None
    error = None
    stopped = True
    window_open = False
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        window_open = True
        cv2.resizeWindow(title, 1280, 592)
        reader.start()
        while True:
            snapshot = reader.snapshot()
            source_error = snapshot.error
            if snapshot.finished and (live or snapshot.frame is None):
                source_error = source_error or (
                    "相机在保存前停止" if live else "离线输入没有可用帧"
                )
            if source_error:
                if frozen is None:
                    raise RuntimeError(source_error)
                error = source_error
            latest = snapshot.frame
            displayed = frozen if frozen is not None else latest
            state = (
                "FROZEN"
                if frozen is not None
                else "LIVE" if live else "REPLAY_END" if snapshot.finished else "REPLAY"
            )
            message = (
                f"Source failed: {source_error}. Enter saves the frozen copy; q exits."
                if source_error
                else "Inspect the frozen RGB and depth; Enter saves this exact frame."
                if frozen is not None
                else "Keep the object unobstructed; space freezes the displayed frame."
            )
            cv2.imshow(title, panel(displayed, state, message, cv2, np))
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                window_open = False
                break
            if key in (27, ord("q")):
                break
            if key == 32:
                if frozen is not None:
                    if not source_error:
                        frozen = None
                        frozen_ns = None
                elif latest is not None:
                    # Freeze RGB, depth, K and clock together so later frames cannot change the saved evidence.
                    frozen = replace(
                        latest,
                        rgb=latest.rgb.copy(),
                        depth_m=latest.depth_m.copy(),
                        K=latest.K.copy(),
                    )
                    frozen_ns = time.monotonic_ns()
            elif key in (10, 13) and frozen is not None:
                saved = save_frame(
                    output,
                    frozen,
                    frozen_ns=frozen_ns,
                    confirmed_ns=time.monotonic_ns(),
                    mode="live_realsense" if live else "offline_replay",
                )
                break
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
    except Exception as failure:
        message = f"{type(failure).__name__}: {failure}"
        error = f"{error}; {message}" if error else message
    finally:
        deadline = time.monotonic() + 15.0
        close = reader.close(timeout_s=0)
        while close.pending and time.monotonic() < deadline:
            if window_open:
                try:
                    cv2.imshow(
                        title,
                        panel(
                            frozen if frozen is not None else latest,
                            "STOPPING",
                            "Waiting for the RGB-D source to close...",
                            cv2,
                            np,
                        ),
                    )
                    cv2.waitKey(20)
                except Exception:
                    window_open = False
            close = reader.close(timeout_s=0.02)
        stopped = close.stopped
        if close.error and close.error != error:
            error = f"{error}; {close.error}" if error else close.error
        if not stopped:
            error = f"{error}; source shutdown timeout" if error else "source shutdown timeout"
        if window_open:
            try:
                cv2.destroyWindow(title)
            except Exception as failure:
                error = (
                    f"{error}; window cleanup: {failure}" if error else f"window cleanup: {failure}"
                )
        write_json(
            output / "result.json",
            {
                "status": "failed" if error else "saved" if saved is not None else "cancelled",
                "capture_saved": saved is not None,
                "source_stopped": stopped,
                "error": error,
                "source_id": saved["source_id"] if saved is not None else None,
                "finished_host_monotonic_ns": time.monotonic_ns(),
            },
        )
    if error:
        raise RuntimeError(f"{error}；已有产物和结束状态保留在 {output}")
    if saved is None:
        print(f"已取消，没有保存物体帧：{output}")
    else:
        print(f"已保存同一来源的 RGB-D：{output / 'frame.npz'}")
        print(
            f"后续建模：cd {shlex.quote(str(args.workspace))} && "
            + shlex.join(
                [
                    sys.executable,
                    "-m",
                    "cordvip_geometry.cli.build_object",
                    "segment",
                    "--rgbd",
                    str(output / "frame.npz"),
                    "--text",
                    "OBJECT.",
                    "--output-dir",
                    str(args.workspace / "reports/cordvip_demo/object_segmentation"),
                ]
            )
        )
    return 0


def main(argv=None, *, workspace=None):
    args = parse_args(argv, workspace=workspace)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

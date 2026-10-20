#!/usr/bin/env python3
"""只读 Raw 或已保存 RGB-D，预览图像和实测深度点，不加载感知模型。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from ..io import new_output


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path, help="只读现有 Raw episode")
    source.add_argument("--rgbd-dir", type=Path, help="回放本示例保存的六位编号 RGB-D NPZ")
    parser.add_argument(
        "--headless", action="store_true", help="不创建窗口，保存最后一帧 RGB-D 预览"
    )
    parser.add_argument("--no-3d", action="store_true", help="只显示 RGB-D，不加载 Open3D viewer")
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示处理全部输入")
    parser.add_argument("--output", type=Path, help="新的 reports/ 子目录；默认按时间命名")
    parser.add_argument("--save-inputs", action="store_true", help="保存实际处理的原网格 RGB-D")
    parser.add_argument(
        "--save-video", action="store_true", help="保存 RGB-D 预览视频，时间戳以 records.jsonl 为准"
    )
    parser.add_argument(
        "--video-fps", type=float, default=10.0, help="输出视频显示帧率，不代表采样速率"
    )
    parser.add_argument("--display-stride", type=int, default=3, help="仅实测三维显示点的像素步长")
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames 必须非负")
    if not args.headless and not args.no_3d and args.display_stride < 1:
        parser.error("--display-stride 必须为正")
    if args.save_video and not (0 < args.video_fps < 1000):
        parser.error("--video-fps 必须有限且位于 (0,1000)")
    if (
        not args.headless
        and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        parser.error("无显示环境；请使用 --headless")
    return args


def run(args):
    import numpy as np
    import cv2
    from .. import geometry
    from ..rgbd import iter_rgbd_directory, rectify_frame, save_frame
    from ..integrations.dexmani import iter_raw_episode
    from ..viewer import DemoViewer, make_panel

    output = new_output(
        args.output, workspace=args.workspace, protected=(args.episode, args.rgbd_dir)
    )
    print(f"Output: {output}", flush=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": config,
                "coordinate_frame": "color_camera",
                "units": "m",
                "rgb_source": "decoded_raw_video" if args.episode else "rgbd_input",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.save_inputs:
        (output / "inputs").mkdir()
    source = viewer = writer = None
    last_preview = None
    processed = 0
    rectification_maps = geometry.RectificationMaps()
    completion = dict(status="failed")
    try:
        if args.episode:
            source = iter_raw_episode(args.episode)
        else:
            source = iter_rgbd_directory(args.rgbd_dir)
        if not args.headless:
            viewer = DemoViewer(show_3d=not args.no_3d)
        live_panel = viewer is not None or args.save_video
        paused = False
        with (output / "records.jsonl").open("x") as records:
            for frame in source:
                rgb, depth_m, K, _ = rectify_frame(frame, maps=rectification_maps)
                index = processed
                if args.save_inputs:
                    save_frame(output / "inputs" / f"{index:06d}.npz", frame)
                started_ns = time.monotonic_ns()
                operation = "observed"
                observed = None
                if viewer is not None and not args.no_3d:
                    display_mask = np.zeros(depth_m.shape, dtype=bool)
                    display_mask[:: args.display_stride, :: args.display_stride] = True
                    observed = geometry.deproject_depth(depth_m, K, mask=display_mask)
                processing_ms = (time.monotonic_ns() - started_ns) / 1e6
                short_source = frame.source_id.rsplit("/", 1)[-1]
                lines = [
                    f"mode=observed frame={index} operation={operation} compute={processing_ms:.1f} ms",
                    f"source={short_source}  clock={frame.clock}",
                ]
                lines.append("Observed RGB-D only; camera frame, meters")
                panel = make_panel(rgb, depth_m, lines=lines) if live_panel else None
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output / "overlay.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.video_fps,
                            (panel.shape[1], panel.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("无法打开视频编码器")
                    writer.write(panel)
                # Offline host clocks belong to the recording, not to this playback process.
                record = dict(
                    index=index,
                    row_index=frame.row_index,
                    source_id=frame.source_id,
                    timestamp_ns=frame.timestamp_ns,
                    clock=frame.clock,
                    operation=operation,
                    processing_ms=processing_ms,
                    source_age_ms=None,
                )
                records.write(json.dumps(record, allow_nan=False) + "\n")
                records.flush()
                processed += 1
                # Retain one completed frame, not an input sequence. Headless runs
                # without video defer the only panel construction until shutdown.
                last_preview = panel if live_panel else (rgb, depth_m, lines)
                quit_requested = False
                if viewer is not None:
                    while True:
                        key = viewer.update(panel, observed, paused=paused)
                        if key in (ord("q"), 27):
                            quit_requested = True
                            break
                        if key == ord(" "):
                            paused = not paused
                        if key == ord("n"):
                            paused = True
                            break
                        if key == ord("s"):
                            stamp = time.monotonic_ns()
                            cv2.imwrite(str(output / f"snapshot_{index:06d}_{stamp}.png"), panel)
                            viewer.screenshot(output / f"cloud_{index:06d}_{stamp}.png")
                        if not paused:
                            break
                        time.sleep(0.01)
                if quit_requested or (args.max_frames and processed >= args.max_frames):
                    break
        if processed == 0:
            raise ValueError("输入序列为空")
        completion["status"] = "completed"
    except BaseException as error:
        completion["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        completion["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors = []
        preview_error = None
        try:
            if last_preview is not None:
                if live_panel:
                    panel = last_preview
                else:
                    rgb, depth_m, lines = last_preview
                    panel = make_panel(rgb, depth_m, lines=lines)
                if not cv2.imwrite(str(output / "last_frame.png"), panel):
                    raise RuntimeError("无法保存 last_frame.png")
        except BaseException as error:
            preview_error = error
            completion["status"] = "failed"
            completion["preview_error"] = f"{type(error).__name__}: {error}"
        finally:
            for name, resource, operation in (
                ("input", source, "close"),
                ("video", writer, "release"),
                ("viewer", viewer, "close"),
            ):
                if resource is not None:
                    try:
                        getattr(resource, operation)()
                    except Exception as error:
                        cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        if cleanup_errors:
            completion["status"] = "failed"
            completion["cleanup_errors"] = cleanup_errors
        completion["processed_frames"] = processed
        try:
            (output / "result.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False))
        except Exception as error:
            message = f"result.json: {type(error).__name__}: {error}"
            cleanup_errors.append(message)
            print(message, file=sys.stderr)
        if preview_error is not None and not active_exception:
            raise preview_error
        if cleanup_errors and not active_exception:
            raise RuntimeError("; ".join(cleanup_errors))
    print(f"完成 {processed} 帧；模式=observed；结果：{output}", flush=True)
    return 0


def main(argv=None, *, workspace=None):
    args = parse_args(argv, workspace=workspace)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("示例已停止。", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"示例失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

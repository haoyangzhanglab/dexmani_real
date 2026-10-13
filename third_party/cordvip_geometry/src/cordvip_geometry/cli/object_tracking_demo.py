#!/usr/bin/env python3
"""离线 RGB-D 顺序回放：实测点云或 FoundationPose；实时追踪使用 track 子命令。"""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import sys
import time


from ..io import _asset, _mask, _new_output, _save_input, _save_ply


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path, help="只读现有 Raw episode")
    source.add_argument("--rgbd-dir", type=Path, help="回放本示例保存的六位编号 RGB-D NPZ")
    parser.add_argument("--mode", choices=("observed", "track"), default="observed")
    parser.add_argument("--asset", type=Path, help="prepare 子命令生成的米制资产目录")
    parser.add_argument("--foundationpose-root", type=Path)
    parser.add_argument("--mask", type=Path, help="起始帧物体 mask，PNG 或 bool/uint8 NPY")
    parser.add_argument("--mask-grid", choices=("source", "rectified"), default="source")
    parser.add_argument("--headless", action="store_true", help="不创建窗口，保存最后一帧投影")
    parser.add_argument("--no-3d", action="store_true", help="只显示 RGB-D，不加载 Open3D viewer")
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示处理全部输入")
    parser.add_argument("--output", type=Path, help="新的 reports/ 子目录；默认按时间命名")
    parser.add_argument("--save-inputs", action="store_true", help="保存实际处理的原网格 RGB-D")
    parser.add_argument("--save-ply", action="store_true", help="逐帧保存相机坐标系模型点云")
    parser.add_argument("--save-video", action="store_true", help="保存投影视频，时间戳以 records.jsonl 为准")
    parser.add_argument("--video-fps", type=float, default=10.0, help="输出视频显示帧率，不代表采样速率")
    parser.add_argument("--display-stride", type=int, default=3, help="仅实测三维显示点的像素步长")
    args = parser.parse_args(argv)
    if args.max_frames < 0 or args.display_stride < 1:
        parser.error("--max-frames 必须非负，--display-stride 必须为正")
    if not (0 < args.video_fps < 1000):
        parser.error("--video-fps 必须有限且位于 (0,1000)")
    if args.mode == "track":
        if args.asset is None or args.foundationpose_root is None:
            parser.error("track 需要 --asset 和 --foundationpose-root")
        if args.headless and args.mask is None:
            parser.error("headless track 需要起始帧 --mask")
    elif any(value is not None for value in (args.asset, args.foundationpose_root, args.mask)):
        parser.error("资产/权重/mask 仅用于显式 --mode track")
    if not args.headless and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("无显示环境；请使用 --headless")
    return args


def _load_modules():
    import numpy as np
    import cv2
    from .. import geometry
    from ..rgbd import iter_rgbd_directory
    from ..integrations.dexmani import iter_raw_episode
    from ..viewer import DemoViewer, make_panel, select_mask

    return np, cv2, geometry, iter_raw_episode, iter_rgbd_directory, DemoViewer, make_panel, select_mask


def run(args):
    np, cv2, geometry, raw_source, directory_source, DemoViewer, make_panel, select_mask = _load_modules()
    tracker = points_local = asset_metadata = None
    bounds_local = None
    if args.mode == "track":
        mesh, points_local, asset_metadata = _asset(args.asset, geometry)
        bounds_local = np.array(list(product(*zip(mesh.bounds[0], mesh.bounds[1]))))
        from ..foundationpose import FoundationPoseTracker

    output = _new_output(args.output, workspace=args.workspace, protected=(args.episode, args.rgbd_dir, args.asset))
    print(f"Output: {output}", flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / "run.json").write_text(json.dumps({
        "config": config, "asset": asset_metadata, "coordinate_frame": "color_camera",
        "units": "m", "pose_quality": "not_ground_truth; no calibrated confidence",
        "rgb_source": "decoded_raw_video" if args.episode else "rgbd_input",
    }, indent=2, ensure_ascii=False))
    if args.save_inputs:
        (output / "inputs").mkdir()
    if args.save_ply:
        (output / "model_clouds").mkdir()
    source = viewer = writer = None
    processed = 0
    pose = None
    pose_source_id = None
    panel = None
    completion = dict(status="failed", inference_performed=False)
    try:
        if args.mode == "track":
            tracker = FoundationPoseTracker(args.foundationpose_root, mesh,
                                           debug_dir=output / "foundationpose")
        if args.episode:
            source = raw_source(args.episode)
        else:
            source = directory_source(args.rgbd_dir)
        if not args.headless:
            viewer = DemoViewer(show_3d=not args.no_3d)
        paused = False
        with (output / "records.jsonl").open("x") as records:
            for frame in source:
                model = frame.distortion_model.removeprefix("distortion.")
                if model not in {"none", "brown_conrady"}:
                    raise ValueError(f"不支持的畸变模型: {frame.distortion_model}")
                coefficients = np.asarray(frame.distortion_coeffs)
                if model == "none" and np.any(coefficients != 0):
                    raise ValueError("none distortion 必须提供零畸变系数")
                initial_mask = None
                if tracker is not None and pose is None and args.mask:
                    initial_mask = _mask(args.mask, frame.depth_m.shape, np, cv2)
                rgb, depth_m, K, rectified_mask = geometry.rectify_rgbd(
                    frame.rgb, frame.depth_m, frame.K, coefficients,
                    mask=initial_mask if args.mask_grid == "source" else None,
                )
                if args.mask_grid == "rectified":
                    rectified_mask = initial_mask
                index = processed
                if args.save_inputs:
                    _save_input(output / "inputs" / f"{index:06d}.npz", frame, np)
                started_ns = time.monotonic_ns()
                operation = "observed"
                if tracker is not None:
                    if pose is None:
                        mask = rectified_mask if rectified_mask is not None else select_mask(rgb)
                        np.save(output / f"mask_{index:06d}_rectified.npy", mask, allow_pickle=False)
                        (output / f"registration_{index:06d}.json").write_text(json.dumps({
                            "source_id": frame.source_id, "input_index": index,
                            "mask_grid": "rectified", "K": K.tolist(),
                        }, indent=2))
                        started_ns = time.monotonic_ns()
                        pose = tracker.register(rgb, depth_m, K, mask)
                        operation = "register"
                    elif frame.source_id != pose_source_id:
                        pose = tracker.track(rgb, depth_m, K)
                        operation = "track"
                    else:
                        operation = "reuse_same_source"
                    pose_source_id = frame.source_id
                    completion["inference_performed"] = True
                completed_ns = time.monotonic_ns()
                processing_ms = (completed_ns - started_ns) / 1e6
                model_points = geometry.transform_points(points_local, pose) if pose is not None else None
                display_mask = np.zeros(depth_m.shape, dtype=bool)
                display_mask[::args.display_stride, ::args.display_stride] = True
                observed = geometry.deproject_depth(depth_m, K, mask=display_mask)
                short_source = frame.source_id.rsplit("/", 1)[-1]
                lines = [f"mode={args.mode} frame={index} operation={operation} compute={processing_ms:.1f} ms",
                         f"source={short_source}  clock={frame.clock}"]
                if tracker is not None:
                    lines.append("gray=observed; cyan=model prediction; pose accuracy is NOT verified")
                else:
                    lines.append("Observed RGB-D only; camera frame, meters")
                bounds_camera = geometry.transform_points(bounds_local, pose) if pose is not None else None
                panel = make_panel(rgb, depth_m, K, model_points=model_points, pose=pose,
                                   bounds_camera=bounds_camera, lines=lines)
                if not cv2.imwrite(str(output / "last_frame.png"), panel):
                    raise RuntimeError("无法保存 last_frame.png")
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.video_fps, (panel.shape[1], panel.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("无法打开视频编码器")
                    writer.write(panel)
                if args.save_ply and model_points is not None:
                    _save_ply(output / "model_clouds" / f"{index:06d}.ply", model_points, np)
                # Offline host clocks belong to the recording, not to this playback process.
                record = dict(index=index, row_index=frame.row_index, source_id=frame.source_id,
                              timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                              operation=operation, processing_ms=processing_ms, source_age_ms=None,
                              pose_camera_from_object=pose.tolist() if pose is not None else None,
                              pose_quality_verified=False)
                records.write(json.dumps(record, allow_nan=False) + "\n")
                records.flush()
                processed += 1
                quit_requested = False
                if viewer is not None:
                    while True:
                        key = viewer.update(panel, observed, model_points, pose=pose, paused=paused)
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
                        if key == ord("r") and tracker is not None:
                            if args.mask:
                                print("下一输入帧将人工圈选 mask；不重用旧帧 mask。", flush=True)
                            args.mask = None
                            pose = None
                            pose_source_id = None
                            paused = False
                            break
                        if not paused:
                            break
                        time.sleep(.01)
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
        for name, resource, operation in (("input", source, "close"),
                                          ("video", writer, "release"),
                                          ("viewer", viewer, "close")):
            if resource is not None:
                try:
                    getattr(resource, operation)()
                except Exception as error:
                    cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        if cleanup_errors:
            completion["status"] = "failed"
            completion["cleanup_errors"] = cleanup_errors
        completion["processed_frames"] = processed
        (output / "result.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False))
        if cleanup_errors and not active_exception:
            raise RuntimeError("; ".join(cleanup_errors))
    print(f"完成 {processed} 帧；模式={args.mode}；结果：{output}", flush=True)
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

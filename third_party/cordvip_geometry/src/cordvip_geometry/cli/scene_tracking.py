"""单物体与多物体 RGB-D 追踪；默认连接 RealSense，离线源只读。"""

import argparse
import json
import math
from pathlib import Path

# Bind Python's Expat before Mesa/native libraries load the system version.
import pyexpat
import re
import sys
import time


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--asset", type=Path, help="reconstruct_object.py 或 prepare 生成的米制物体资产目录"
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="多物体 JSON：objects 中逐个声明 id/role/asset/initial_selection/prompt",
    )
    parser.add_argument(
        "--all-frames", action="store_true", help="离线模式逐帧处理，不按回放速率丢帧或等待"
    )
    parser.add_argument("--max-frames", type=int, help="最多处理的帧数；用于显式的部分验收")
    parser.add_argument(
        "--foundationpose-root", type=Path, default=workspace / "third_party/FoundationPose"
    )
    parser.add_argument(
        "--object-prompt", help='Grounded-SAM2 文本目标，例如 "red mug."；省略时手动圈选'
    )
    from ..sam2_config import add_sam_arguments

    add_sam_arguments(parser, workspace)
    parser.add_argument(
        "--check", action="store_true", help="只检查环境，不加载模型、创建窗口或连接相机"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--episode", type=Path, help="用只读 Raw 替代相机，按 replay-fps 演示异步流程"
    )
    source.add_argument("--rgbd-dir", type=Path, help="用 RGB-D NPZ 替代相机")
    parser.add_argument("--replay-fps", type=float, default=30)
    parser.add_argument("--serial", help="RealSense serial；省略时由 SDK 选择相机")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-3d", action="store_true", help="仅显示双 RGB 窗口；仍生成两类物体点云"
    )
    parser.add_argument(
        "--initial-selection", type=Path, help="复用已确认的 select 目录；必须与离线源首帧完全对应"
    )
    parser.add_argument(
        "--headless", action="store_true", help="无窗口离线回放；需要 --initial-selection"
    )
    parser.add_argument(
        "--with-hand", action="store_true", help="读取同一 Raw 行的实测关节，附加 XHand FK 点云"
    )
    parser.add_argument("--experiment-config", type=Path, help="手安装变换使用的当前实验配置 YAML")
    parser.add_argument("--hand-points", type=int, default=1024, help="固定手表面采样点数")
    parser.add_argument("--output", type=Path, help="尚不存在的新输出目录")
    parser.add_argument(
        "--save-inputs", action="store_true", help="保存所有送入推理的源 RGB-D；注册帧总是保存"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="逐次保存实际处理结果到 frames/；不补齐跳过的源帧",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="保存处理帧叠加视频；播放率为 replay-fps"
    )
    parser.add_argument("--max-age-ms", type=float, default=500, help="源帧最大有效年龄，毫秒")
    parser.add_argument("--pixel-stride", type=int, default=2, help="物体实测点云的像素采样步长")
    parser.add_argument("--depth-tolerance-m", type=float, default=0.015)
    parser.add_argument("--min-matched-pixels", type=int, default=30)
    parser.add_argument("--min-matched-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    if not args.check and args.asset is None and args.scene is None:
        parser.error("追踪需要 --asset 或 --scene；只检查环境可用 --check")
    if args.scene and (args.asset or args.initial_selection or args.object_prompt):
        parser.error("--scene 内逐个指定资产、选择和 prompt，不能混用单物体参数")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames 必须为正整数")
    if args.serial and (args.episode or args.rgbd_dir):
        parser.error("--serial 只用于实时相机")
    offline = args.episode is not None or args.rgbd_dir is not None
    if args.all_frames and not offline:
        parser.error("--all-frames 仅用于离线源")
    if args.initial_selection is not None and not offline:
        parser.error("--initial-selection 仅用于离线源，不能将历史 mask 用于新的相机帧")
    if args.headless and (not offline or (args.initial_selection is None and args.scene is None)):
        parser.error("--headless 需要离线源和 --initial-selection")
    if args.with_hand and args.episode is None:
        parser.error("--with-hand 需要 --episode，以同一 Raw 行的实测关节生成手点")
    if args.experiment_config is not None and not args.with_hand:
        parser.error("--experiment-config 需要 --with-hand")
    if args.object_prompt is not None and not args.object_prompt.strip():
        parser.error("--object-prompt 不能为空")
    if not offline and min(args.width, args.height, args.fps) < 1:
        parser.error("实时相机 width、height、fps 必须为正整数")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride 必须为正整数")
    if args.with_hand and args.hand_points < 1:
        parser.error("--hand-points 必须为正整数")
    if not math.isfinite(args.max_age_ms) or args.max_age_ms <= 0:
        parser.error("--max-age-ms 必须为有限正数")
    if (args.save_video or (offline and not args.all_frames)) and (
        not math.isfinite(args.replay_fps) or args.replay_fps <= 0
    ):
        parser.error("--replay-fps 必须为有限正数")
    return args


def load_scene(path, workspace, *, offline, headless):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or set(data) != {"objects"}
        or not isinstance(data["objects"], list)
    ):
        raise ValueError("scene JSON must contain an objects list")
    if not data["objects"]:
        raise ValueError("scene requires at least one fixed instance")
    objects = {}
    for item in data["objects"]:
        if not isinstance(item, dict) or set(item) - {
            "id",
            "role",
            "asset",
            "initial_selection",
            "prompt",
        }:
            raise ValueError("Each scene object accepts id, role, asset, initial_selection, prompt")
        name = item.get("id", "")
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-z][a-z0-9_]*", name)
            or "__" in name
            or name in objects
        ):
            raise ValueError(
                "Scene object IDs must be unique lowercase identifiers without double underscores"
            )
        item = item.copy()
        item["role"] = item.get("role", "object")
        if item["role"] not in {"object", "reference"}:
            raise ValueError("Scene role must be object or reference")
        for key in ("asset", "initial_selection"):
            if item.get(key) is not None:
                candidate = Path(item[key]).expanduser()
                item[key] = (
                    candidate if candidate.is_absolute() else workspace / candidate
                ).resolve()
                if not item[key].is_dir():
                    raise ValueError(f"{name}: missing {key} directory {item[key]}")
        if item.get("initial_selection") and not offline:
            raise ValueError("Historical initial_selection cannot initialize a new camera frame")
        if headless and not item.get("initial_selection"):
            raise ValueError(f"{name}: headless scene requires an exact-source initial_selection")
        if item.get("prompt") is not None and (
            not isinstance(item["prompt"], str) or not item["prompt"].strip()
        ):
            raise ValueError("Scene prompt must be nonempty text")
        objects[name] = item
    return objects


def choose_masks(objects, frame, args, viewer, output, worker):
    import numpy as np
    from ..io import load_initial_selection, write_confirmed_selection, write_json
    from ..rgbd import rectify_frame, save_frame

    masks, details, replay_objects = {}, {}, []
    try:
        for name, item in objects.items():
            candidates = []
            detection_metrics = None
            if item.get("initial_selection"):
                masks[name], details[name] = load_initial_selection(
                    item["initial_selection"], frame
                )
                replay_objects.append({**item, "initial_selection": str(item["initial_selection"])})
                continue
            # Confirm on the original RGB grid so the same source-verified selection can be reused.
            viewer.begin_selection(frame.rgb, manual=not item.get("prompt"))
            if item.get("prompt"):
                if not worker.submit(
                    "predict",
                    frame,
                    item["prompt"],
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                ):
                    raise RuntimeError("Scene worker was not ready for selection")

                def selecting():
                    key = viewer.update(
                        frame.rgb,
                        frame.rgb,
                        lines=[f"Grounded-SAM2: finding {name}"],
                        status_text="Text segmentation",
                        help_text="q: cancel",
                    )
                    if key in (27, ord("q")):
                        raise KeyboardInterrupt("Selection cancelled")

                result = wait_worker(worker, selecting)
                if result.frame is not frame or result.operation != "predict":
                    raise RuntimeError("Rejected mismatched detection result")
                candidates = result.value
                detection_metrics = dict(worker.segmenter.last_metrics)
                viewer.set_candidates(candidates)
                if not candidates:
                    viewer.use_polygon()
            while True:
                key = viewer.update(
                    frame.rgb,
                    frame.rgb,
                    lines=[f'Select {name} ({item["role"]}), then Enter. q cancels the scene.'],
                    status_text="Confirm every instance on the same frozen frame",
                    help_text="1-9 / [ ] / click: candidate | p: polygon | Enter: confirm | q: quit",
                )
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt("Scene selection cancelled")
                if key == ord("p"):
                    viewer.use_polygon()
                viewer.choose_candidate(key)
                if key in (10, 13):
                    mask = viewer.selected_mask()
                    if mask is not None and np.count_nonzero(mask & (frame.depth_m > 0)) >= 4:
                        _, selected_depth, _, selected_mask = rectify_frame(frame, mask)
                        if np.count_nonzero(selected_mask & (selected_depth > 0)) < 4:
                            continue
                        masks[name] = selected_mask
                        details[name] = dict(
                            method="gui_confirmed",
                            prompt=item.get("prompt"),
                            selected_instance=viewer.selected_candidate_index,
                            detection_metrics=detection_metrics,
                        )
                        selected_index = viewer.selected_candidate_index
                        candidate = (
                            candidates[selected_index] if selected_index is not None else None
                        )
                        scores = dict(
                            detection_score=candidate["score"] if candidate is not None else None,
                            mask_score=candidate["mask_score"] if candidate is not None else None,
                            score_kind="grounding_dino" if candidate is not None else None,
                            mask_score_kind="sam2_predicted_iou" if candidate is not None else None,
                            box_threshold=args.box_threshold,
                            text_threshold=args.text_threshold,
                        )
                        details[name].update(scores)
                        selected = output / "initial_selections" / name
                        selected.mkdir(parents=True)
                        save_frame(selected / "source_rgbd.npz", frame)
                        write_confirmed_selection(
                            selected,
                            mask,
                            dict(
                                selection="gui_confirmed",
                                selected_instance=viewer.selected_candidate_index,
                                mask_grid="original_rgb_grid; not_automatically_rectified",
                                source_id=frame.source_id,
                                timestamp_ns=frame.timestamp_ns,
                                clock=frame.clock,
                                prompt=item.get("prompt"),
                                detection_metrics=detection_metrics,
                                **scores,
                                segmentation_model=(
                                    worker.metadata["segmentation"] if item.get("prompt") else None
                                ),
                            ),
                            mask_file="mask.npy",
                        )
                        replay_objects.append({**item, "initial_selection": str(selected)})
                        break
            viewer.cancel_selection()
        save_frame(output / "initial_source_rgbd.npz", frame)
        for name, mask in masks.items():
            path = output / f"initial_{name}_rectified.npy"
            np.save(path, mask, allow_pickle=False)
        write_json(
            output / "initialization.json",
            dict(
                source_id=frame.source_id,
                timestamp_ns=frame.timestamp_ns,
                clock=frame.clock,
                row_index=frame.row_index,
                objects=details,
            ),
        )
        write_json(
            output / "replay_scene.json",
            {
                "objects": [
                    {k: str(v) if isinstance(v, Path) else v for k, v in item.items()}
                    for item in replay_objects
                ]
            },
        )
        return masks
    finally:
        if viewer is not None:
            viewer.cancel_selection()


def wait_worker(worker, heartbeat, *, loading=False):
    while True:
        status = worker.snapshot()
        if status.error:
            raise RuntimeError(status.error)
        if loading and status.ready:
            return
        if not loading:
            result = worker.poll()
            if result is not None:
                if result.error:
                    raise RuntimeError(result.error)
                return result
        heartbeat()
        time.sleep(0.005)


def scene_display(result, colors, hand=None):
    import cv2
    import numpy as np
    from ..geometry import project_points

    overlay = result["rgb"].copy()
    observed, models, observed_colors, model_colors = [], [], [], []
    for name, item in result["objects"].items():
        color = colors[name]
        mask = item["cloud"].semantic_mask
        overlay[mask] = (overlay[mask] * 0.62 + np.asarray(color) * 0.38).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
        points = item["cloud"].observed_points_camera
        observed.append(points)
        observed_colors.append(np.tile(np.asarray(color) / 255.0, (len(points), 1)))
        points = item["model_points_camera"]
        models.append(points)
        model_colors.append(np.tile(np.asarray(color) / 255.0, (len(points), 1)))
        if item["pose_valid"]:
            p = item["pose"]
            axes = np.array([[0, 0, 0], [0.035, 0, 0], [0, 0.035, 0], [0, 0, 0.035]])
            uv, good = project_points(axes @ p[:3, :3].T + p[:3, 3], result["K"])
            if good.all() and np.abs(uv).max() < 1e6:
                uv = np.rint(uv).astype(int)
                for i, axis_color in enumerate(((255, 70, 70), (70, 255, 70), (70, 70, 255)), 1):
                    cv2.line(overlay, tuple(uv[0]), tuple(uv[i]), axis_color, 2)
                cv2.putText(overlay, name, tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if hand is not None:
        uv, good = project_points(hand["points_camera"], result["K"])
        h, w = overlay.shape[:2]
        good &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for u, v in np.rint(uv[good]).astype(int):
            cv2.circle(overlay, (int(u), int(v)), 1, (255, 140, 35), -1)
    return overlay, dict(
        observed_points=np.concatenate(observed),
        model_points=np.concatenate(models),
        observed_colors=np.concatenate(observed_colors),
        model_colors=np.concatenate(model_colors),
    )


class SceneTrackingSession:
    """Own scene acquisition, one model worker, publication and viewer lifetime."""

    def __init__(self, args):
        self.args = args

    def run(self):
        from ..sam2_config import check_sam_environment
        from .check_tracking import check_tracking_environment, print_tracking_report

        self.live = not (self.args.episode or self.args.rgbd_dir)
        self.objects = (
            load_scene(
                self.args.scene,
                self.args.workspace,
                offline=not self.live,
                headless=self.args.headless,
            )
            if self.args.scene
            else {
                "object": dict(
                    id="object",
                    role="object",
                    asset=self.args.asset,
                    initial_selection=self.args.initial_selection,
                    prompt=self.args.object_prompt,
                )
            }
        )
        need_pose = self.args.scene is None or any(
            item.get("asset") for item in self.objects.values()
        )
        need_grounding = any(
            item.get("prompt") and not item.get("initial_selection")
            for item in self.objects.values()
        )
        if need_grounding:
            for name in ("box_threshold", "text_threshold"):
                if not 0 < getattr(self.args, name) < 1:
                    raise ValueError(f"{name} 必须位于 (0,1)")
        if need_pose:
            if (
                not math.isfinite(self.args.depth_tolerance_m)
                or self.args.depth_tolerance_m <= 0
                or self.args.min_matched_pixels < 1
            ):
                raise ValueError("位姿深度容差必须为有限正数，匹配像素数必须为正整数")
            if not 0 < self.args.min_matched_fraction <= 1:
                raise ValueError("--min-matched-fraction 必须位于 (0,1]")
        report = check_tracking_environment(
            self.args.foundationpose_root,
            need_camera=self.live,
            need_gui=not self.args.check and not self.args.headless,
            need_3d_viewer=not self.args.headless and not self.args.no_3d,
            need_pose=need_pose,
        )
        report["checks"].extend(
            check_sam_environment(
                self.args,
                require_grounding=need_grounding,
            )
        )
        report["ok"] = all(item["ok"] for item in report["checks"])
        print_tracking_report(report)
        if self.args.check or not report["ok"]:
            return 0 if report["ok"] else 2

        import cv2
        import numpy as np
        from ..io import load_asset, new_output, write_json, write_observation
        from ..integrations.dexmani import (
            iter_raw_episode,
            iter_live_realsense,
            RawHandCloudProvider,
        )
        from ..live_workers import LatestFrameReader, SceneWorker
        from ..rgbd import iter_rgbd_directory, rectify_frame, save_frame
        from ..geometry import RectificationMaps
        from ..scene_tracking import SceneTracker, scene_payload
        from ..segmentation import GroundedSAM2Segmenter

        self.rectification_maps = RectificationMaps()
        self.assets, self.asset_metadata = {}, {}
        for name, item in self.objects.items():
            if item.get("asset"):
                mesh, points, normals, metadata = load_asset(item["asset"])
                self.assets[name] = (mesh, points, normals)
                self.asset_metadata[name] = metadata
        protected = [self.args.scene, self.args.episode, self.args.rgbd_dir]
        for item in self.objects.values():
            protected.extend((item.get("asset"), item.get("initial_selection")))
        self.output = new_output(
            self.args.output, workspace=self.args.workspace, protected=protected
        )
        if self.args.save_frames:
            (self.output / "frames").mkdir()
        if self.args.save_inputs:
            (self.output / "inputs").mkdir()
        self.roles = {name: item["role"] for name, item in self.objects.items()}
        palette = [(50, 220, 230), (220, 90, 230), (80, 220, 100), (240, 220, 70)]
        self.colors = {name: palette[i % len(palette)] for i, name in enumerate(self.objects)}
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self.args).items()
        }
        self.hand_provider = (
            RawHandCloudProvider(
                self.args.episode,
                config_path=self.args.experiment_config,
                num_points=self.args.hand_points,
            )
            if self.args.with_hand
            else None
        )
        write_json(
            self.output / "run.json",
            dict(
                config=config,
                scene={
                    n: {k: str(v) if isinstance(v, Path) else v for k, v in o.items()}
                    for n, o in self.objects.items()
                },
                assets=self.asset_metadata,
                environment_check=report,
                coordinate_frame="color_camera",
                units="m",
                colors_rgb=self.colors,
                observed_semantics="current semantic mask + measured depth, independent of mesh; segmentation can be wrong",
                model_semantics="fixed generated surface + per-instance T_camera_object; geometry agreement is not confidence",
                processing=(
                    "all frames, unpaced"
                    if self.args.all_frames
                    else "latest frame, bounded mailbox, no backlog"
                ),
                source_clock="preserved; offline timestamps do not prove sensor synchronization",
                hand=self.hand_provider.metadata if self.hand_provider else None,
                video="processed frames only, replay-fps playback, no reconstruction of original timing",
            ),
        )
        if self.args.episode:
            source_factory = lambda: iter_raw_episode(self.args.episode)
        elif self.args.rgbd_dir:
            source_factory = lambda: iter_rgbd_directory(self.args.rgbd_dir)
        else:
            source_factory = lambda: iter_live_realsense(
                serial=self.args.serial,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
            )
        self.viewer = self.worker = self.segmenter = self.reader = self.source = self.video = None
        self.current = self.hand = self.overlay = self.display = None
        self.records = (self.output / "records.jsonl").open("x")
        processed = source_count = 0
        last_version = 0
        failure = None
        completed = False
        first_row = last_row = None
        self.published_state = "LOADING"

        print(f"Scene output: {self.output}", flush=True)
        try:
            write_observation(
                self.output / "latest.npz",
                scene_payload(
                    None, self.roles, valid=False, state="LOADING", max_age_ms=self.args.max_age_ms
                ),
            )
            if not self.args.headless:
                from ..live_viewer import LiveViewer

                self.viewer = LiveViewer(show_3d=not self.args.no_3d)
            self.segmenter = GroundedSAM2Segmenter(
                self.args.sam_python,
                self.args.sam_root,
                self.args.sam_checkpoint,
                sam_config=self.args.sam_config,
                dino_weights=self.args.dino_weights,
                log_path=self.output / "sam2_worker.log",
            )
            self.worker = SceneWorker(
                self.segmenter,
                lambda segmenter: SceneTracker(
                    segmenter,
                    self.assets,
                    foundationpose_root=self.args.foundationpose_root,
                    debug_dir=self.output / "foundationpose",
                    pixel_stride=self.args.pixel_stride,
                    depth_tolerance_m=self.args.depth_tolerance_m,
                    min_matched_pixels=self.args.min_matched_pixels,
                    min_matched_fraction=self.args.min_matched_fraction,
                ),
            ).start()
            wait_worker(self.worker, self.heartbeat, loading=True)
            (self.output / "models.json").write_text(json.dumps(self.worker.metadata, indent=2))
            if self.args.all_frames:
                self.source = iter(source_factory())
            else:
                self.reader = LatestFrameReader(
                    source_factory,
                    replay_interval_s=0 if self.live else 1 / self.args.replay_fps,
                    wait_on_first=not self.live,
                ).start()
            while True:
                if self.source is not None:
                    frame = next(self.source, None)
                    if frame is None:
                        completed = True
                        break
                    source_count += 1
                else:
                    snapshot = self.reader.snapshot()
                    source_count = snapshot.version
                    if snapshot.error:
                        raise RuntimeError(snapshot.error)
                    if snapshot.version == last_version:
                        if snapshot.finished:
                            if self.live:
                                raise RuntimeError("Live RGB-D source ended")
                            completed = True
                            break
                        self.heartbeat()
                        time.sleep(0.002)
                        continue
                    frame, last_version = snapshot.frame, snapshot.version
                rectified = rectify_frame(frame, maps=self.rectification_maps)[:3]
                initial = (
                    choose_masks(
                        self.objects, frame, self.args, self.viewer, self.output, self.worker
                    )
                    if processed == 0
                    else None
                )
                if processed == 0:
                    # First-frame text prompts may have lazily loaded the detector.
                    (self.output / "models.json").write_text(
                        json.dumps(self.worker.metadata, indent=2)
                    )
                if not self.worker.submit("step", frame, *rectified, initial_masks=initial):
                    raise RuntimeError("Scene worker was not ready for a frame")
                finished = wait_worker(self.worker, self.heartbeat)
                if finished.frame is not frame or finished.operation != "step":
                    raise RuntimeError("Rejected mismatched scene worker result")
                self.current = finished.value
                self.hand = self.hand_provider.for_frame(frame) if self.hand_provider else None
                processed += 1
                if processed == 1:
                    first_row = frame.row_index
                last_row = frame.row_index
                if self.reader is not None and processed == 1:
                    self.reader.resume()
                age_ms = (time.monotonic_ns() - frame.timestamp_ns) / 1e6 if self.live else None
                valid = not self.live or (0 <= age_ms <= self.args.max_age_ms)
                state = (
                    "TRACKING" if self.live and valid else "REPLAY" if not self.live else "STALE"
                )
                payload = scene_payload(
                    self.current,
                    self.roles,
                    valid=valid,
                    state=state,
                    max_age_ms=self.args.max_age_ms,
                    hand=self.hand,
                )
                write_observation(self.output / "latest.npz", payload)
                self.published_state = state
                if self.args.save_frames:
                    write_observation(self.output / "frames" / f"{processed-1:06d}.npz", payload)
                if self.args.save_inputs:
                    save_frame(self.output / "inputs" / f"{processed-1:06d}.npz", frame)
                record = dict(
                    source_id=frame.source_id,
                    row_index=frame.row_index,
                    timestamp_ns=frame.timestamp_ns,
                    clock=frame.clock,
                    operation=self.current["operation"],
                    source_age_ms=age_ms,
                    segmentation_ms=self.current["segmentation_ms"],
                    pose_ms=self.current["pose_ms"],
                    segmentation_metrics=self.current["segmentation_metrics"],
                    processing_ms=(self.current["completed_ns"] - self.current["started_ns"]) / 1e6,
                    completed_ns=self.current["completed_ns"],
                    valid=valid,
                    objects={
                        name: dict(
                            state=item["state"],
                            error=item["error"],
                            pose_valid=item["pose_valid"],
                            observed_valid=item["observed_valid"],
                            T_camera_object=item["pose"].tolist() if item["pose_valid"] else None,
                            diagnostic_T_camera_object=(
                                item["diagnostic_pose"].tolist()
                                if item["diagnostic_pose"] is not None
                                else None
                            ),
                            diagnostics=item["cloud"].diagnostics,
                        )
                        for name, item in self.current["objects"].items()
                    },
                )
                self.records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.records.flush()
                if self.viewer is not None or self.args.save_video:
                    self.overlay, self.display = scene_display(self.current, self.colors, self.hand)
                    lines = [
                        f"{state} | source row={frame.row_index} | processed={processed} | "
                        f'pipeline={record["processing_ms"]:.1f} ms | SAM={record["segmentation_ms"]:.1f} ms'
                    ]
                    lines.extend(
                        f'{name}: {item["state"]} | observed={len(item["cloud"].observed_points_camera)}'
                        for name, item in self.current["objects"].items()
                    )
                    lines.append("Model agreement is not pose ground truth; orange=FK hand")
                    if self.args.save_video:
                        panel = cv2.cvtColor(
                            np.hstack((self.current["rgb"], self.overlay)), cv2.COLOR_RGB2BGR
                        )
                        cv2.rectangle(
                            panel, (0, 0), (panel.shape[1], 20 + 20 * len(lines)), (0, 0, 0), -1
                        )
                        for i, line in enumerate(lines):
                            cv2.putText(
                                panel,
                                line,
                                (8, 18 + 20 * i),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,
                                (255, 255, 255),
                                1,
                            )
                        if self.video is None:
                            self.video = cv2.VideoWriter(
                                str(self.output / "overlay.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                self.args.replay_fps,
                                (panel.shape[1], panel.shape[0]),
                            )
                            if not self.video.isOpened():
                                raise RuntimeError("Could not create scene overlay video")
                        self.video.write(panel)
                    if self.viewer is not None:
                        key = self.viewer.update(
                            self.current["rgb"],
                            self.overlay,
                            **self.display,
                            result_valid=valid,
                            hand_points=self.hand["points_camera"] if self.hand else None,
                            lines=lines,
                            status_text="Per-instance pose and observed-depth states shown below",
                            help_text="v: measured/model layers | s: screenshot | q: quit",
                        )
                        if key == ord("s"):
                            self.viewer.screenshot(self.output / f"scene_{processed:06d}.png")
                        if key in (27, ord("q")):
                            break
                if self.args.max_frames and processed >= self.args.max_frames:
                    break
            if processed == 0:
                raise ValueError("RGB-D source contained no frames")
        except KeyboardInterrupt:
            print("Scene stopped by user", flush=True)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            print(failure, flush=True)
        finally:
            cleanup_errors = []
            cleanup = {}
            invalidation_method = None
            # Invalidate before resource shutdown, even after partial inference.
            try:
                write_observation(
                    self.output / "latest.npz",
                    scene_payload(
                        self.current,
                        self.roles,
                        valid=False,
                        state="ERROR" if failure else "STOPPED",
                        max_age_ms=self.args.max_age_ms,
                    ),
                )
                invalidation_method = "invalid_write"
            except Exception as error:
                cleanup_errors.append(f"invalidation: {type(error).__name__}: {error}")
                try:
                    (self.output / "latest.npz").unlink(missing_ok=True)
                    invalidation_method = "removed"
                except Exception as error:
                    cleanup_errors.append(f"invalidation unlink: {type(error).__name__}: {error}")
                    cleanup_errors.append("已发布结果失效未确认 (latest.npz invalidation unconfirmed)")
            if self.viewer is not None and processed:
                try:
                    self.viewer.screenshot(self.output / "last_frame.png")
                except Exception as error:
                    cleanup_errors.append(f"screenshot: {type(error).__name__}: {error}")
            for name, resource in (
                ("reader", self.reader),
                ("source", self.source),
                ("worker", self.worker),
                ("segmenter", self.segmenter if self.worker is None else None),
                ("viewer", self.viewer),
                ("video", self.video),
                ("records", self.records),
            ):
                if resource is None:
                    continue
                try:
                    if name in ("reader", "worker"):
                        status = resource.close(timeout_s=2)
                        cleanup[name] = dict(
                            status=(
                                "unconfirmed"
                                if not status.stopped
                                else "failed" if status.error else "closed"
                            ),
                            stopped=status.stopped,
                            error=status.error,
                        )
                        if not status.stopped or status.error:
                            cleanup_errors.append(f"cleanup {name}: {status}")
                    else:
                        if name == "video":
                            resource.release()
                        else:
                            resource.close()
                        cleanup[name] = dict(status="closed")
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    cleanup[name] = dict(status="failed", error=message)
                    cleanup_errors.append(f"cleanup {name}: {message}")
            cuda_allocated = cuda_reserved = None
            try:
                if self.assets:
                    import torch

                    if torch.cuda.is_available():
                        cuda_allocated = torch.cuda.max_memory_allocated()
                        cuda_reserved = torch.cuda.max_memory_reserved()
            except Exception as error:
                cleanup_errors.append(f"CUDA diagnostics: {type(error).__name__}: {error}")
            failure = "; ".join(filter(None, [failure, *cleanup_errors])) or None
            try:
                (self.output / "result.json").write_text(
                    json.dumps(
                        dict(
                            status=(
                                "failed"
                                if failure
                                else "completed_replay" if completed and not self.live else "stopped"
                            ),
                            error=failure,
                            cleanup_errors=cleanup_errors,
                            cleanup=cleanup,
                            latest_invalidation_confirmed=invalidation_method is not None,
                            latest_invalidation_method=invalidation_method,
                            processed_frames=processed,
                            source_frames=source_count,
                            first_processed_row=first_row,
                            last_processed_row=last_row,
                            live_camera_requested=self.live,
                            cuda_peak_allocated_bytes=cuda_allocated,
                            cuda_peak_reserved_bytes=cuda_reserved,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            except Exception as error:
                failure = "; ".join(filter(None, [failure, f"result.json: {type(error).__name__}: {error}"]))
                print(f"Scene result publication failed: {failure}", file=sys.stderr, flush=True)
        return 1 if failure else 0

    def heartbeat(self):
        from ..io import write_observation
        from ..scene_tracking import scene_payload

        fresh = self.current is not None
        if self.current is not None and self.live:
            age = (time.monotonic_ns() - self.current["frame"].timestamp_ns) / 1e6
            fresh = 0 <= age <= self.args.max_age_ms
            if not fresh and self.published_state != "STALE":
                write_observation(
                    self.output / "latest.npz",
                    scene_payload(
                        self.current,
                        self.roles,
                        valid=False,
                        state="STALE",
                        max_age_ms=self.args.max_age_ms,
                    ),
                )
                self.published_state = "STALE"
        if self.viewer is not None:
            key = self.viewer.update(
                self.current["rgb"] if self.current else None,
                self.overlay,
                **(self.display or {}),
                result_valid=fresh,
                hand_points=self.hand["points_camera"] if self.hand else None,
                lines=["Loading / processing / waiting for the next frame"],
                status_text=self.published_state,
                help_text="q: quit",
            )
            if key in (27, ord("q")):
                raise KeyboardInterrupt("Viewer closed")


def main(argv=None, *, workspace=None):
    try:
        return SceneTrackingSession(parse_args(argv, workspace=workspace)).run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

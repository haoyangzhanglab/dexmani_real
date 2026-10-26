"""单物体与多物体 RGB-D 追踪；默认连接 RealSense，离线源只读。"""

import json
import math

# 先绑定 Python 的 Expat，避免 Mesa 等原生库抢先加载系统版本。
import pyexpat  # noqa: F401
import sys
import time
from pathlib import Path

from .arguments import parse_tracking_args
from .scene_initialization import choose_masks, load_scene


class SceneTrackingSession:
    """持有场景源、模型线程与输出，退出前先使已发布结果失效。"""

    def __init__(self, args):
        self.args = args
        self.viewer = self.worker = self.segmenter = self.reader = self.source = self.video = None
        self.records = None
        self.current = self.hand = self.overlay = self.display = None
        self.processed = self.source_count = self.last_version = 0
        self.first_row = self.last_row = None
        self.failure = None
        self.completed = False
        self.published_state = "LOADING"

    def run(self):
        report = self.check_environment()
        if self.args.check or not report["ok"]:
            return 0 if report["ok"] else 2
        self.prepare_output(report)
        print(f"Scene output: {self.output}", flush=True)
        try:
            self.start()
            self.run_frames()
        except KeyboardInterrupt:
            print("Scene stopped by user", flush=True)
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            print(self.failure, flush=True)
        finally:
            self.close()
        return 1 if self.failure else 0

    def check_environment(self):
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
        return report

    def prepare_output(self, report):
        from ..geometry import RectificationMaps
        from ..integrations.dexmani import RawHandCloudProvider
        from ..io import load_asset, new_output, write_json

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
        self.records = (self.output / "records.jsonl").open("x")

    def start(self):
        from ..integrations.dexmani import iter_live_realsense, iter_raw_episode
        from ..live_workers import LatestFrameReader, SceneWorker, wait_worker
        from ..rgbd import iter_rgbd_directory
        from ..scene_tracking import SceneTracker
        from ..segmentation import GroundedSAM2Segmenter

        self.publish_latest(valid=False, state="LOADING")
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
        if self.args.all_frames:
            self.source = iter(source_factory())
        else:
            self.reader = LatestFrameReader(
                source_factory,
                replay_interval_s=0 if self.live else 1 / self.args.replay_fps,
                wait_on_first=not self.live,
            ).start()

    def run_frames(self):
        from ..live_workers import wait_worker
        from ..rgbd import rectify_frame

        while True:
            if self.source is not None:
                frame = next(self.source, None)
                if frame is None:
                    self.completed = True
                    break
                self.source_count += 1
            else:
                snapshot = self.reader.snapshot()
                self.source_count = snapshot.version
                if snapshot.error:
                    raise RuntimeError(snapshot.error)
                if snapshot.version == self.last_version:
                    if snapshot.finished:
                        if self.live:
                            raise RuntimeError("Live RGB-D source ended")
                        self.completed = True
                        break
                    self.heartbeat()
                    time.sleep(0.002)
                    continue
                frame, self.last_version = snapshot.frame, snapshot.version
            rectified = rectify_frame(frame, maps=self.rectification_maps)[:3]
            initial = (
                choose_masks(
                    self.objects, frame, self.args, self.viewer, self.output, self.worker
                )
                if self.processed == 0
                else None
            )
            if self.processed == 0:
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
            self.processed += 1
            if self.processed == 1:
                self.first_row = frame.row_index
            self.last_row = frame.row_index
            if self.reader is not None and self.processed == 1:
                self.reader.resume()
            age_ms = (time.monotonic_ns() - frame.timestamp_ns) / 1e6 if self.live else None
            valid = not self.live or (0 <= age_ms <= self.args.max_age_ms)
            state = (
                "TRACKING" if self.live and valid else "REPLAY" if not self.live else "STALE"
            )
            record = self.publish_frame(frame, age_ms, valid, state)
            if self.update_display(frame, valid, state, record):
                break
            if self.args.max_frames and self.processed >= self.args.max_frames:
                break
        if self.processed == 0:
            raise ValueError("RGB-D source contained no frames")

    def publish_latest(self, *, valid, state):
        from ..io import write_observation
        from ..scene_tracking import scene_payload

        payload = scene_payload(
            self.current, self.roles, valid=valid, state=state,
            max_age_ms=self.args.max_age_ms, hand=self.hand,
        )
        write_observation(self.output / "latest.npz", payload)
        self.published_state = state
        return payload

    def publish_frame(self, frame, age_ms, valid, state):
        from ..io import write_observation
        from ..rgbd import save_frame

        payload = self.publish_latest(valid=valid, state=state)
        if self.args.save_frames:
            write_observation(self.output / "frames" / f"{self.processed-1:06d}.npz", payload)
        if self.args.save_inputs:
            save_frame(self.output / "inputs" / f"{self.processed-1:06d}.npz", frame)
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
        return record

    def update_display(self, frame, valid, state, record):
        if self.viewer is None and not self.args.save_video:
            return False
        import cv2
        import numpy as np

        from ..viewer import scene_display

        self.overlay, self.display = scene_display(
            self.current, self.colors, self.hand,
            include_clouds=self.viewer is not None and not self.args.no_3d,
        )
        lines = [
            f"{state} | source row={frame.row_index} | processed={self.processed} | "
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
                self.viewer.screenshot(self.output / f"scene_{self.processed:06d}.png")
            if key in (27, ord("q")):
                return True
        return False

    def close(self):
        cleanup_errors = []
        cleanup = {}
        invalidation_method = None
        # Invalidate before resource shutdown, even after partial inference.
        try:
            self.publish_latest(valid=False, state="ERROR" if self.failure else "STOPPED")
            invalidation_method = "invalid_write"
        except Exception as error:
            cleanup_errors.append(f"invalidation: {type(error).__name__}: {error}")
            try:
                (self.output / "latest.npz").unlink(missing_ok=True)
                invalidation_method = "removed"
            except Exception as error:
                cleanup_errors.append(f"invalidation unlink: {type(error).__name__}: {error}")
                cleanup_errors.append("已发布结果失效未确认 (latest.npz invalidation unconfirmed)")
        if self.viewer is not None and self.processed:
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
        self.failure = "; ".join(filter(None, [self.failure, *cleanup_errors])) or None
        try:
            (self.output / "result.json").write_text(
                json.dumps(
                    dict(
                        status=(
                            "failed"
                            if self.failure
                            else "completed_replay" if self.completed and not self.live else "stopped"
                        ),
                        error=self.failure,
                        cleanup_errors=cleanup_errors,
                        cleanup=cleanup,
                        latest_invalidation_confirmed=invalidation_method is not None,
                        latest_invalidation_method=invalidation_method,
                        processed_frames=self.processed,
                        source_frames=self.source_count,
                        first_processed_row=self.first_row,
                        last_processed_row=self.last_row,
                        live_camera_requested=self.live,
                        cuda_peak_allocated_bytes=cuda_allocated,
                        cuda_peak_reserved_bytes=cuda_reserved,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as error:
            self.failure = "; ".join(filter(None, [self.failure, f"result.json: {type(error).__name__}: {error}"]))
            print(f"Scene result publication failed: {self.failure}", file=sys.stderr, flush=True)

    def heartbeat(self):
        fresh = self.current is not None
        if self.current is not None and self.live:
            age = (time.monotonic_ns() - self.current["frame"].timestamp_ns) / 1e6
            fresh = 0 <= age <= self.args.max_age_ms
            if not fresh and self.published_state != "STALE":
                self.publish_latest(valid=False, state="STALE")
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
        return SceneTrackingSession(parse_tracking_args(argv, workspace=workspace)).run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

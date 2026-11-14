"""结合可见轮廓与实测深度的多物体位姿跟踪。"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .camera import RGBDRectifier, camera_rays
from .foundationpose import (
    MIN_DEPTH_PIXELS,
    FoundationPoseTracker,
    TrackingFailure,
    track_objects,
)
from .mesh import MeshRaycaster, load_mesh

logger = logging.getLogger(__name__)
PIXEL_STRIDE = 2


@dataclass(frozen=True)
class TrackingConfig:
    depth_tolerance_m: float = 0.015
    min_depth_pixels: int = 30
    min_depth_fraction: float = 0.5
    min_precision: float = 0.5
    min_recall: float = 0.5

    segmentation_interval_frames: int = 2

    recovery_enabled: bool = True
    recovery_retry_interval_frames: int = 15
    recovery_box_threshold: float = 0.3
    recovery_text_threshold: float = 0.25

    def __post_init__(self):
        if (
            type(self.segmentation_interval_frames) is not int
            or self.segmentation_interval_frames < 1
        ):
            raise ValueError("segmentation_interval_frames must be a positive integer")
        if type(self.recovery_enabled) is not bool:
            raise ValueError("recovery_enabled must be bool")
        if (
            type(self.recovery_retry_interval_frames) is not int
            or self.recovery_retry_interval_frames < 1
        ):
            raise ValueError("recovery_retry_interval_frames must be positive integer")
        for value in (self.recovery_box_threshold, self.recovery_text_threshold):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("recovery thresholds must lie in (0, 1]")
        if not np.isfinite(self.depth_tolerance_m) or self.depth_tolerance_m <= 0:
            raise ValueError("depth_tolerance_m must be finite and positive")
        if type(self.min_depth_pixels) is not int or self.min_depth_pixels < 1:
            raise ValueError("min_depth_pixels must be a positive integer")
        for value in (self.min_depth_fraction, self.min_precision, self.min_recall):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("quality fractions must lie in (0, 1]")


@dataclass
class SceneResult:
    """仅 poses 中的对象具有当前有效位姿；其余对象由 valid/failures 标记。"""

    poses: dict[str, np.ndarray]
    quality: dict[str, dict[str, float | int | None]]
    valid: dict[str, bool]
    failures: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


def pose_quality(mask, observed, rendered, config):
    """前景遮挡只能由 mask 外的有效传感器深度支持。"""
    valid = np.isfinite(observed) & (observed >= 0.001)
    projected = np.isfinite(rendered) & (rendered > 0)
    depth_error = np.abs(observed - rendered)
    hidden = (
        projected & ~mask & valid & (observed < rendered - config.depth_tolerance_m)
    )
    visible = projected & ~hidden
    intersection = mask & visible
    depth_target = mask & valid
    matched = depth_target & projected & (depth_error <= config.depth_tolerance_m)
    count = int(matched.sum())
    residual = depth_error[depth_target & projected]
    mask_pixels = int(mask.sum())
    visible_pixels = int(visible.sum())
    intersection_pixels = int(intersection.sum())
    metrics = dict(
        mask_pixels=mask_pixels,
        visible_pixels=visible_pixels,
        precision=intersection_pixels / max(1, visible_pixels),
        recall=intersection_pixels / max(1, mask_pixels),
        iou=float(intersection_pixels / max(1, (mask | visible).sum())),
        depth_pixels=count,
        depth_fraction=float(count / max(1, depth_target.sum())),
        median_depth_error_m=float(np.median(residual)) if len(residual) else None,
        occluded_fraction=float(hidden.sum() / max(1, projected.sum())),
    )
    accepted = (
        count >= config.min_depth_pixels
        and metrics["depth_fraction"] >= config.min_depth_fraction
        and metrics["precision"] >= config.min_precision
        and metrics["recall"] >= config.min_recall
    )
    return metrics, accepted


def pose_depth_quality(observed, rendered, config):
    """没有当前 mask 时，所有方向的有效深度偏差都计为不匹配。"""
    projected = np.isfinite(rendered) & (rendered > 0)
    support = projected & np.isfinite(observed) & (observed >= 0.001)
    residual = np.abs(observed[support] - rendered[support])
    count = int(np.count_nonzero(residual <= config.depth_tolerance_m))
    fraction = count / max(1, len(residual))
    metrics: dict[str, float | int | None] = dict.fromkeys(
        (
            "mask_pixels",
            "visible_pixels",
            "precision",
            "recall",
            "iou",
            "occluded_fraction",
        ),
        -1,
    )
    metrics.update(
        depth_pixels=count,
        depth_fraction=fraction,
        median_depth_error_m=float(np.median(residual)) if len(residual) else None,
    )
    return (
        metrics,
        count >= config.min_depth_pixels and fraction >= config.min_depth_fraction,
    )


class SceneTracker:
    def __init__(self, mesh_paths, *, workspace, config=TrackingConfig()):
        import torch

        from .segmentation import SAM2Segmenter
        from .weights import DEFAULT_MODELS_DIR, resolve_sam_model

        self.config = config
        workspace = Path(workspace)
        models_dir = workspace / DEFAULT_MODELS_DIR
        checkpoint, sam_config = resolve_sam_model(models_dir)
        self.segmenter = SAM2Segmenter(
            workspace / "third_party/Grounded-SAM-2",
            checkpoint,
            sam_config,
            models_dir / "grounding-dino-base",
        )
        self.segmenter.load_grounding()
        self.trackers = {}
        self.renderers = {}
        shared = None
        for name, path in mesh_paths.items():
            tracker = FoundationPoseTracker(
                workspace / "third_party/FoundationPose",
                load_mesh(path),
                shared_with=shared,
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = MeshRaycaster(tracker.mesh)

        # 在 worker 宣告就绪前预热固定尺寸网络；不创建视频历史或物体位姿。
        with torch.inference_mode():
            image_size = self.segmenter.tracker.image_size
            image = torch.zeros((1, 3, image_size, image_size), device="cuda").to(
                memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.segmenter.tracker.forward_image(image)
            refiner = next(iter(self.trackers.values())).estimator.refiner
            inputs = torch.zeros(
                (len(self.trackers), 6, *refiner.cfg.input_resize), device="cuda"
            )
            with (
                torch.backends.cudnn.flags(enabled=False),
                torch.autocast("cuda", dtype=torch.float16),
            ):
                refiner.model(inputs, inputs)
            torch.cuda.synchronize()

    def initialize(self, frame, prompts):
        self.prompts = {
            name: prompt.strip().lower().rstrip(". ")
            for name, prompt in prompts.items()
        }
        normalized = list(self.prompts.values())
        if (
            set(prompts) != set(self.trackers)
            or not all(normalized)
            or len(set(normalized)) != len(normalized)
        ):
            raise ValueError("Prompts must uniquely identify every object")
        self.states = dict.fromkeys(self.trackers, "TRACKING")
        self.next_retry = dict.fromkeys(self.trackers, 0)
        self.frame_index = 0
        self.last_sam_frame = 0
        self.frame_signature = (
            frame.depth_m.shape,
            frame.K.copy(),
            frame.distortion_model,
            np.asarray(frame.distortion_coeffs).copy(),
        )
        self.frame_cache = {}
        for tracker in self.trackers.values():
            tracker.estimator.pose_last = None

        if frame.distortion_model.removeprefix("distortion.") not in {
            "none",
            "brown_conrady",
        }:
            raise ValueError(f"Unsupported distortion model: {frame.distortion_model}")
        self.rectifier = RGBDRectifier(
            frame.depth_m.shape, frame.K, frame.distortion_coeffs
        )
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        self.directions = camera_rays(depth_m.shape, frame.K, PIXEL_STRIDE)

        try:
            candidates = self.segmenter.detect(
                rgb, ". ".join(self.prompts.values()) + "."
            )
        except RuntimeError as error:
            raise TrackingFailure(f"Initial detection: {error}") from error
        boxes = {}
        for name in self.trackers:
            try:
                boxes[name] = self._match_box(candidates, self.prompts[name])
            except TrackingFailure as error:
                raise TrackingFailure(f"{name}: {error}") from error

        try:
            masks = self.segmenter.initialize(rgb, boxes)
        except RuntimeError as error:
            raise TrackingFailure(f"Initial segmentation: {error}") from error
        result = self._estimate(rgb, depth_m, frame.K, masks, first=True)
        logger.debug("HOI initialization timings_ms=%s", result.timings_ms)
        if not all(result.valid.values()):
            raise TrackingFailure(str(result.failures))
        return result

    def track(self, frame):
        shape, K, model, coeffs = self.frame_signature
        if (
            frame.depth_m.shape != shape
            or not np.array_equal(frame.K, K)
            or frame.distortion_model != model
            or not np.array_equal(frame.distortion_coeffs, coeffs)
        ):
            raise ValueError("Camera parameters changed within episode")
        if frame.rgb.shape != (*shape, 3):
            raise ValueError("RGB grid changed within episode")
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        self.frame_index += 1
        result = self._estimate(rgb, depth_m, frame.K, None, first=False)
        logger.debug("HOI frame=%d timings_ms=%s", self.frame_index, result.timings_ms)
        return result

    @staticmethod
    def _match_box(candidates, prompt):
        matches = [
            candidate
            for candidate in candidates
            if candidate["label"].strip().lower().rstrip(". ") == prompt
        ]
        if len(matches) != 1:
            raise TrackingFailure(
                f"expected one detection for {prompt!r}, got {len(matches)}"
            )
        return matches[0]["box_xyxy"]

    def _check_depth(self, mask, depth):
        valid = np.isfinite(depth) & (depth >= 0.001)
        if (
            mask is None
            or np.count_nonzero((mask & valid)[::PIXEL_STRIDE, ::PIXEL_STRIDE])
            < self.config.min_depth_pixels
            or np.count_nonzero(mask & valid) < MIN_DEPTH_PIXELS
        ):
            raise TrackingFailure("insufficient segmented depth")

    def _refresh_masks(self, rgb, result, previous):
        started = perf_counter()
        try:
            masks = self.segmenter.track(rgb)
        except RuntimeError:
            # 事件刷新可能在 refinement 之后发生；共享状态错误仍须终止实例。
            for name, tracker in self.trackers.items():
                tracker.estimator.pose_last = previous[name]
            raise
        result.timings_ms["sam2_tracking"] = (perf_counter() - started) * 1000
        self.last_sam_frame = self.frame_index
        return masks

    def _estimate(self, rgb, depth_m, K, masks, *, first):
        # 后端只替换 pose_last，不原地修改旧张量；保留引用即可回滚居中网格位姿。
        previous = {
            name: tracker.estimator.pose_last for name, tracker in self.trackers.items()
        }
        result = SceneResult(
            poses={},
            quality={name: {} for name in self.trackers},
            valid=dict.fromkeys(self.trackers, False),
        )
        # 只恢复进入本帧前已丢失的对象，新丢失对象从下一帧开始尝试。
        due = [
            name
            for name in self.trackers
            if self.states[name] == "LOST" and self.frame_index >= self.next_retry[name]
        ]

        result.timings_ms["sam2_tracking"] = 0.0
        if not first and (
            self.frame_index - self.last_sam_frame
            >= self.config.segmentation_interval_frames
            or (due and self.config.recovery_enabled)
        ):
            masks = self._refresh_masks(rgb, result, previous)
        sampled = depth_m[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        rendered_depths = {}

        def quality(name, pose, mask):
            if np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
                raise TrackingFailure("FoundationPose returned nonfinite/invalid pose")
            if name not in rendered_depths:
                rendered_depths[name] = (
                    self.renderers[name]
                    .render(self.directions.reshape(-1, 3), pose)
                    .reshape(sampled.shape)
                )
            rendered = rendered_depths[name]
            if mask is None:
                metrics, accepted = pose_depth_quality(sampled, rendered, self.config)
            else:
                metrics, accepted = pose_quality(
                    mask[::PIXEL_STRIDE, ::PIXEL_STRIDE], sampled, rendered, self.config
                )
            metrics.update(
                mask_source=int(mask is not None),
                mask_age_frames=self.frame_index - self.last_sam_frame,
            )
            return metrics, accepted

        def reject(name, error):
            self.trackers[name].estimator.pose_last = previous[name]
            result.poses.pop(name, None)
            result.valid[name] = False
            result.failures[name] = str(error)
            logger.warning("%s: %s", name, error)

        def accept(name, pose, mask):
            metrics, accepted = quality(name, pose, mask)
            result.quality[name] = metrics
            if not accepted:
                raise TrackingFailure(
                    f"Semantic depth/model disagreement; quality={metrics}"
                )
            result.poses[name] = pose
            result.valid[name] = True

        def register(name, mask):
            try:
                pose = self.trackers[name].register(rgb, depth_m, K, mask)
            except RuntimeError as error:
                reject(name, error)
                return False
            try:
                accept(name, pose, mask)
            except TrackingFailure as error:
                reject(name, error)
                return False
            return True

        active = {}
        for name, tracker in self.trackers.items():
            if self.states[name] == "LOST":
                continue
            try:
                if masks is not None:
                    self._check_depth(masks.get(name), depth_m)
                active[name] = tracker
            except TrackingFailure as error:
                reject(name, error)
        started = perf_counter()
        sam_before = result.timings_ms["sam2_tracking"]
        if first:
            for name in active:
                register(name, masks[name])
        elif active:
            try:
                candidates = track_objects(active, rgb, depth_m, K, self.frame_cache)
            except RuntimeError:
                logger.warning(
                    "FoundationPose batch failed; isolating objects", exc_info=True
                )
                # 批量失败后回滚，再隔离实际失败对象；健康路径仍只推理一次。
                candidates = {}
                for name, tracker in active.items():
                    tracker.estimator.pose_last = previous[name]
                for name, tracker in active.items():
                    try:
                        candidates.update(
                            track_objects(
                                {name: tracker}, rgb, depth_m, K, self.frame_cache
                            )
                        )
                    except RuntimeError as error:
                        reject(name, error)
            if masks is None:
                refresh = False
                for name, pose in list(candidates.items()):
                    try:
                        metrics, accepted = quality(name, pose, None)
                        result.quality[name] = metrics
                        refresh |= not accepted
                    except TrackingFailure as error:
                        reject(name, error)
                        del candidates[name]
                if refresh:
                    masks = self._refresh_masks(rgb, result, previous)
                    for name in list(candidates):
                        try:
                            self._check_depth(masks.get(name), depth_m)
                        except TrackingFailure as error:
                            reject(name, error)
                            del candidates[name]
            for name, pose in candidates.items():
                if masks is None:
                    # 所有候选已通过深度门控；直接接受，保留刚计算的质量。
                    result.poses[name] = pose
                    result.valid[name] = True
                    continue
                try:
                    accept(name, pose, masks[name])
                except TrackingFailure as error:
                    reject(name, error)
        result.timings_ms["registration" if first else "foundationpose_tracking"] = (
            perf_counter() - started
        ) * 1000 - (result.timings_ms["sam2_tracking"] - sam_before)

        if due and self.config.recovery_enabled and not first:
            recovery_start = perf_counter()
            for name in due:
                self.next_retry[name] = (
                    self.frame_index + self.config.recovery_retry_interval_frames
                )
            try:
                started = perf_counter()
                candidates = self.segmenter.detect(
                    rgb,
                    ". ".join(self.prompts[n] for n in due) + ".",
                    self.config.recovery_box_threshold,
                    self.config.recovery_text_threshold,
                )
                result.timings_ms["recovery_detection"] = (
                    perf_counter() - started
                ) * 1000
                boxes = {}
                for name in due:
                    try:
                        boxes[name] = self._match_box(candidates, self.prompts[name])
                    except TrackingFailure as error:
                        reject(name, error)
                started = perf_counter()
                recovered_masks = (
                    self.segmenter.segment_boxes(rgb, boxes) if boxes else {}
                )
                result.timings_ms["recovery_segmentation"] = (
                    perf_counter() - started
                ) * 1000
            except RuntimeError as error:
                for name in due:
                    reject(name, error)
                recovered_masks = {}
            corrections = {}
            started = perf_counter()
            for name, mask in recovered_masks.items():
                try:
                    self._check_depth(mask, depth_m)
                except TrackingFailure as error:
                    reject(name, error)
                    continue
                if register(name, mask):
                    corrections[name] = mask
            result.timings_ms["recovery_registration"] = (
                perf_counter() - started
            ) * 1000
            if corrections:
                try:
                    masks = self.segmenter.correct_masks(corrections)
                except RuntimeError:
                    for name, tracker in self.trackers.items():
                        tracker.estimator.pose_last = previous[name]
                    raise
                for name, pose in list(result.poses.items()):
                    try:
                        self._check_depth(masks.get(name), depth_m)
                        accept(name, pose, masks[name])
                    except TrackingFailure as error:
                        reject(name, error)
            result.timings_ms["recovery_total"] = (
                perf_counter() - recovery_start
            ) * 1000
        # 只在最终 mask 校验后提交状态，失败的恢复尝试保留既定重试时间。
        for name in self.trackers:
            if result.valid[name]:
                self.states[name] = "TRACKING"
            else:
                if self.states[name] == "TRACKING":
                    self.next_retry[name] = self.frame_index + 1
                self.states[name] = "LOST"
                result.failures.setdefault(name, "LOST; awaiting recovery retry")
        result.poses = {
            name: result.poses[name] for name in self.trackers if result.valid[name]
        }
        return result

"""结合可见轮廓与实测深度的多物体位姿跟踪。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import RGBDRectifier, camera_rays
from .foundationpose import (
    MIN_DEPTH_PIXELS,
    FoundationPoseTracker,
    TrackingFailure,
    track_objects,
)
from .mesh import MeshRaycaster, load_mesh

PIXEL_STRIDE = 2


@dataclass(frozen=True)
class TrackingConfig:
    depth_tolerance_m: float = 0.015
    min_depth_pixels: int = 30
    min_depth_fraction: float = 0.5
    min_precision: float = 0.5
    min_recall: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.depth_tolerance_m) or self.depth_tolerance_m <= 0:
            raise ValueError("depth_tolerance_m must be finite and positive")
        if type(self.min_depth_pixels) is not int or self.min_depth_pixels < 1:
            raise ValueError("min_depth_pixels must be a positive integer")
        for value in (self.min_depth_fraction, self.min_precision, self.min_recall):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("quality fractions must lie in (0, 1]")


@dataclass
class SceneResult:
    poses: dict[str, np.ndarray]
    quality: dict[str, dict[str, float | int | None]]


def pose_quality(mask, observed, rendered, config):
    """前景遮挡只能由 mask 外的有效传感器深度支持。"""
    valid = np.isfinite(observed) & (observed >= 0.001)
    projected = np.isfinite(rendered) & (rendered > 0)
    depth_error = np.abs(observed - rendered)
    hidden = projected & ~mask & valid & (observed < rendered - config.depth_tolerance_m)
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

        candidates = self.segmenter.detect(rgb, ". ".join(prompts.values()) + ".")
        boxes = {}
        for name, prompt in prompts.items():
            matches = [
                item
                for item in candidates
                if item["label"].strip().lower().rstrip(".")
                == prompt.strip().lower().rstrip(".")
            ]
            if len(matches) != 1:
                raise TrackingFailure(
                    f"{name}: expected one first-frame detection, got {len(matches)}"
                )
            boxes[name] = matches[0]["box_xyxy"]

        masks = self.segmenter.initialize(rgb, boxes)
        return self._estimate(rgb, depth_m, frame.K, masks, first=True)

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
        masks = self.segmenter.track(rgb)
        return self._estimate(rgb, depth_m, frame.K, masks, first=False)

    def _estimate(self, rgb, depth_m, K, masks, *, first) -> SceneResult:
        """返回物体局部到相机的位姿及质量；任一物体失效则终止本帧。"""
        sampled_depth = depth_m[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        for name in self.trackers:
            mask = masks[name]
            selected = mask[::PIXEL_STRIDE, ::PIXEL_STRIDE] & (sampled_depth > 0)
            sampled_count = np.count_nonzero(selected)
            depth_count = np.count_nonzero(mask & (depth_m >= 0.001))
            if sampled_count < self.config.min_depth_pixels or depth_count < MIN_DEPTH_PIXELS:
                raise TrackingFailure(
                    f"{name}: insufficient segmented depth: "
                    f"{sampled_count} independent points, "
                    f"{depth_count} FoundationPose pixels; requires "
                    f"{self.config.min_depth_pixels} and {MIN_DEPTH_PIXELS}"
                )

        quality = {}
        poses = {}
        if not first:
            poses = track_objects(
                self.trackers, rgb, depth_m, K, self.frame_cache
            )
        for name, tracker in self.trackers.items():
            if first:
                try:
                    poses[name] = tracker.register(rgb, depth_m, K, masks[name])
                except TrackingFailure as error:
                    raise TrackingFailure(f"{name}: {error}") from error

            rendered = (
                self.renderers[name]
                .render(self.directions.reshape(-1, 3), poses[name])
                .reshape(sampled_depth.shape)
            )
            metrics, accepted = pose_quality(
                masks[name][::PIXEL_STRIDE, ::PIXEL_STRIDE], sampled_depth, rendered, self.config
            )
            quality[name] = metrics
            if not accepted:
                raise TrackingFailure(
                    f"{name}: Semantic depth/model disagreement; quality={metrics}"
                )
        return SceneResult(poses, quality)

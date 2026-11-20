"""结合可见轮廓与实测深度的多物体位姿跟踪。"""

import logging
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .camera import RGBDFrame, RGBDRectifier, camera_rays
from .foundationpose import (
    FoundationPoseTracker,
    TrackingFailure,
    bbox_center_offset,
    refinement_xy_delta,
    track_objects,
)
from .kalman import KalmanXY
from .mesh import MeshRaycaster
from .ostrack import OSTrackAdapter, mask_to_xyxy
from .segmentation import SAM2Segmenter, normalize_prompt
from .weights import DEFAULT_MODELS_DIR, GROUNDING_WEIGHTS_DIR, resolve_sam_model

logger = logging.getLogger(__name__)
PIXEL_STRIDE = 4
PIXEL_GRID = np.s_[::PIXEL_STRIDE, ::PIXEL_STRIDE]
# 容纳 float32 迭代误差，拒绝明显非刚体候选。
POSE_ROTATION_ATOL = 1e-3
POSE_HOMOGENEOUS_ATOL = 1e-6


@dataclass(frozen=True)
class TrackingConfig:
    depth_tolerance_m: float = 0.015
    min_depth_pixels: int = 8
    min_depth_fraction: float = 0.5
    min_depth_coverage: float = 0.5
    min_precision: float = 0.5
    min_recall: float = 0.5

    segmentation_interval_frames: int = 10
    max_translation_speed_m_s: float | None = None
    max_rotation_speed_rad_s: float | None = None

    recovery_enabled: bool = True
    recovery_retry_interval_frames: int = 2
    recovery_box_threshold: float = 0.3
    recovery_text_threshold: float = 0.25

    ostrack_checkpoint: str = "third_party/weights/ostrack/OSTrack_ep0300.pth.tar"
    lost_timeout_s: float = 0.1
    xy_prior_gain: float = 0.2
    kalman_xy_prior: bool = False
    recovery_crop: bool = True

    def __post_init__(self):
        if not isinstance(self.ostrack_checkpoint, str) or not self.ostrack_checkpoint.strip():
            raise ValueError("ostrack_checkpoint must be a nonempty path")
        for name in (
            "min_depth_pixels",
            "segmentation_interval_frames",
            "recovery_retry_interval_frames",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "depth_tolerance_m",
            "lost_timeout_s",
            "max_translation_speed_m_s",
            "max_rotation_speed_rad_s",
        ):
            value = getattr(self, name)
            if value is None and name.startswith("max_"):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0 < float(value) < np.inf
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "min_depth_fraction",
            "min_depth_coverage",
            "min_precision",
            "min_recall",
            "recovery_box_threshold",
            "recovery_text_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0 < float(value) <= 1
            ):
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("recovery_enabled", "kalman_xy_prior", "recovery_crop"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if (
            isinstance(self.xy_prior_gain, bool)
            or not isinstance(self.xy_prior_gain, Real)
            or not 0 <= self.xy_prior_gain <= 1
        ):
            raise ValueError("xy_prior_gain must lie in [0, 1]")


@dataclass
class SceneResult:
    """仅 poses 中的对象具有当前有效位姿；其余对象由 valid/failures 标记。"""

    poses: dict[str, np.ndarray]
    quality: dict[str, dict[str, float | int | None]]
    valid: dict[str, bool]
    failures: dict[str, str] = field(default_factory=dict)


def recovery_image_crop(rgb, box):
    height, width = rgb.shape[:2]
    if not np.isfinite(box).all():
        return rgb, (0, 0)
    x1, y1, x2, y2 = np.clip(box, (0, 0, 0, 0), (width, height, width, height))
    if x1 >= x2 or y1 >= y2:
        return rgb, (0, 0)
    side = max(256, int(np.ceil(2 * max(x2 - x1, y2 - y1))))
    crop_width, crop_height = min(side, width), min(side, height)
    left = int(np.clip((x1 + x2 - crop_width) // 2, 0, width - crop_width))
    top = int(np.clip((y1 + y2 - crop_height) // 2, 0, height - crop_height))
    return rgb[top:top + crop_height, left:left + crop_width], (left, top)


def pose_quality(mask, observed, rendered, config, *, coarse_recovery=False):
    """前景遮挡只能由 mask 外的有效传感器深度支持。"""
    valid = np.isfinite(observed) & (observed >= 0.001)
    projected = np.isfinite(rendered) & (rendered > 0)
    projected_count = int(projected.sum())
    depth_error = np.abs(observed - rendered)
    hidden = (
        projected & ~mask & valid & (observed < rendered - config.depth_tolerance_m)
    )
    visible = projected & ~hidden
    intersection = mask & visible
    depth_target = mask & valid
    matched = depth_target & projected & (depth_error <= config.depth_tolerance_m)
    matched_count = int(matched.sum())
    residual = depth_error[depth_target & projected]
    mask_pixels = int(mask.sum())
    visible_pixels = int(visible.sum())
    intersection_pixels = int(intersection.sum())
    depth_support_pixels = int(depth_target.sum())
    visible_depth_pixels = int((visible & valid).sum())
    supported_intersection = int((intersection & valid).sum())
    metrics = {
        "mask_pixels": mask_pixels,
        "visible_pixels": visible_pixels,
        "precision": intersection_pixels / max(1, visible_pixels),
        "recall": intersection_pixels / max(1, mask_pixels),
        "iou": float(intersection_pixels / max(1, (mask | visible).sum())),
        "depth_pixels": matched_count,
        "depth_fraction": matched_count / max(1, depth_support_pixels),
        "depth_coverage": int((projected & valid).sum()) / max(1, projected_count),
        "depth_support_pixels": depth_support_pixels,
        "visible_depth_pixels": visible_depth_pixels,
        "mask_depth_coverage": depth_support_pixels / max(1, mask_pixels),
        "visible_depth_coverage": visible_depth_pixels / max(1, visible_pixels),
        "supported_precision": supported_intersection / max(1, visible_depth_pixels),
        "supported_recall": supported_intersection / max(1, depth_support_pixels),
        "median_depth_error_m": float(np.median(residual)) if len(residual) else None,
        "occluded_fraction": float(hidden.sum() / max(1, projected_count)),
    }
    # 有效深度充分时，才将与 Mask 或可见轮廓的分歧判为 LOST。
    mask_supported = (
        depth_support_pixels >= config.min_depth_pixels
        and metrics["mask_depth_coverage"] >= config.min_depth_coverage
    )
    visible_supported = (
        visible_depth_pixels >= config.min_depth_pixels
        and metrics["visible_depth_coverage"] >= config.min_depth_coverage
    )
    # 当前检测已定位目标，粗注册允许 Mask 中有更多不匹配的深度点。
    min_depth_fraction = config.min_depth_fraction * (0.8 if coarse_recovery else 1.0)
    if (
        mask_supported
        and (
            metrics["depth_fraction"] < min_depth_fraction
            or metrics["supported_recall"] < config.min_recall
        )
    ) or (visible_supported and metrics["supported_precision"] < config.min_precision):
        return metrics, "LOST"
    if (
        mask_supported
        and visible_supported
        and matched_count >= config.min_depth_pixels
        and metrics["precision"] >= config.min_precision
        and metrics["recall"] >= config.min_recall
    ):
        return metrics, "VALID"
    return metrics, "OCCLUDED"


def pose_depth_quality(observed, rendered, config):
    """纯深度验证允许前景遮挡，仍要求 CAD 有足够的直接深度匹配。"""
    projected = np.isfinite(rendered) & (rendered > 0)
    support = projected & np.isfinite(observed) & (observed >= 0.001)
    residual = observed[support] - rendered[support]
    depth_error = np.abs(residual)
    matched_count = int(np.count_nonzero(depth_error <= config.depth_tolerance_m))
    hidden_count = int(np.count_nonzero(residual < -config.depth_tolerance_m))
    support_count = len(residual)
    projected_count = int(projected.sum())
    visible_count = projected_count - hidden_count
    visible_depth_count = support_count - hidden_count
    coverage = support_count / max(1, projected_count)
    visible_coverage = visible_depth_count / max(1, visible_count)
    fraction = matched_count / max(1, visible_depth_count)
    metrics: dict[str, float | int | None] = {
        "mask_pixels": -1,
        "visible_pixels": visible_count,
        "precision": -1,
        "recall": -1,
        "iou": -1,
        "occluded_fraction": hidden_count / max(1, projected_count),
        "depth_pixels": matched_count,
        "depth_fraction": fraction,
        "depth_coverage": coverage,
        "depth_support_pixels": support_count,
        "visible_depth_pixels": visible_depth_count,
        "mask_depth_coverage": -1,
        "visible_depth_coverage": visible_coverage,
        "supported_precision": -1,
        "supported_recall": -1,
        "median_depth_error_m": float(np.median(depth_error)) if support_count else None,
    }
    if visible_depth_count < config.min_depth_pixels or coverage < config.min_depth_coverage:
        return metrics, "OCCLUDED"
    if fraction < config.min_depth_fraction:
        return metrics, "LOST"
    # 实测前景遮挡至少一半投影时，可用深度充分的可见区域作为面积基准。
    match_area = (
        visible_count
        if hidden_count >= visible_count and visible_coverage >= config.min_depth_coverage
        else projected_count
    )
    min_matches = match_area * config.min_depth_coverage * config.min_depth_fraction
    if matched_count < max(config.min_depth_pixels, min_matches):
        return metrics, "OCCLUDED"
    return metrics, "VALID"


def pose_motion(previous, current, tracker, dt, config):
    """在居中网格坐标中比较位姿，避免物体原点偏移引入额外平移。"""
    previous = previous @ tracker.object_from_center
    current = current @ tracker.object_from_center
    displacement = current[:3, 3] - previous[:3, 3]
    translation = np.sqrt(np.sum(displacement * displacement))
    trace = np.einsum("ij,ij->", previous[:3, :3], current[:3, :3])
    rotation = np.arccos(np.clip((trace - 1) / 2, -1, 1), dtype=trace.dtype)
    continuous = True
    # 没有实测上限时只报告变化量；零 dt 也不能据此推断无限速度。
    if dt > 0:
        # 在位姿指标精度下比较阈值，避免标量运算改变临界值的舍入。
        if config.max_translation_speed_m_s is not None:
            translation_limit = translation.dtype.type(
                config.max_translation_speed_m_s * dt + config.depth_tolerance_m
            )
            continuous = translation <= translation_limit
        if config.max_rotation_speed_rad_s is not None:
            angular_slack = 2 * np.arcsin(
                min(1, config.depth_tolerance_m / tracker.estimator.diameter)
            )
            rotation_limit = rotation.dtype.type(
                config.max_rotation_speed_rad_s * dt + angular_slack
            )
            continuous &= rotation <= rotation_limit
    return {
        "translation_delta_m": float(translation),
        "rotation_delta_rad": float(rotation),
        "pose_dt_s": dt,
    }, bool(continuous)


class SceneTracker:
    source_sequence: int
    timestamp_ns: int

    def __init__(
        self,
        mesh_paths,
        *,
        workspace,
        config: TrackingConfig = TrackingConfig(),  # noqa: B008 — 不可变配置可共享。
    ):
        if not mesh_paths:
            raise ValueError("SceneTracker requires at least one object mesh")
        for name, mesh_path in mesh_paths.items():
            if not Path(mesh_path).is_file():
                raise FileNotFoundError(f"{name}: CAD mesh not found: {mesh_path}")
        import torch

        self.config = config
        workspace = Path(workspace)
        self.ostrack = OSTrackAdapter(
            workspace / "third_party/OSTrack",
            workspace / Path(config.ostrack_checkpoint).expanduser(),
        )
        models_dir = workspace / DEFAULT_MODELS_DIR
        checkpoint, sam_config = resolve_sam_model(models_dir)
        self.segmenter = SAM2Segmenter(
            workspace / "third_party/Grounded-SAM-2",
            checkpoint,
            sam_config,
            models_dir / GROUNDING_WEIGHTS_DIR,
        )
        self.segmenter.load_grounding()
        self.trackers = {}
        self.renderers = {}
        shared = None
        for name, path in mesh_paths.items():
            tracker = FoundationPoseTracker(
                workspace / "third_party/FoundationPose",
                path,
                shared_with=shared,
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = MeshRaycaster(tracker.mesh)

        # 预热固定尺寸网络，让 worker 就绪后可直接处理首帧。
        with torch.inference_mode():
            sam_model = self.segmenter.image_predictor.model
            image_size = sam_model.image_size
            image = torch.zeros((1, 3, image_size, image_size), device="cuda").to(
                memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sam_model.forward_image(image)
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

    def initialize(self, frame: RGBDFrame, prompts: dict[str, str]) -> SceneResult:
        if set(prompts) != set(self.trackers):
            raise ValueError("Prompts must cover every object")
        self.prompts = {
            name: normalize_prompt(prompt).removesuffix(".")
            for name, prompt in prompts.items()
        }
        if len(set(self.prompts.values())) != len(self.prompts):
            raise ValueError("Prompts must uniquely identify every object")
        self.next_retry: dict[str, int] = {}
        self.last_pose: dict[str, tuple[np.ndarray, int]] = {}
        self.motion_priors: dict[str, KalmanXY] = {}
        self._set_frame_time(frame, first=True)
        self.ostrack.clear()
        self.bbox_offsets: dict[str, np.ndarray] = {}
        self.last_mask_sequence: dict[str, int] = {}
        self.frame_index = 0
        self.last_sam_frame = 0
        self.frame_signature = (
            frame.K.copy(),
            frame.distortion_model,
            tuple(frame.distortion_coeffs),
        )
        self.frame_cache: dict[str, Any] = {}
        for tracker in self.trackers.values():
            tracker.estimator.pose_last = None

        self.rectifier = RGBDRectifier(
            frame.depth_m.shape, frame.K, frame.distortion_model, frame.distortion_coeffs
        )
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        self.directions = camera_rays(depth_m.shape, frame.K, PIXEL_STRIDE)

        boxes, failures = self._detect_boxes(rgb, self.trackers)
        if failures:
            name, reason = next(iter(failures.items()))
            raise TrackingFailure(f"{name}: {reason}")

        masks = self.segmenter.segment_boxes(rgb, boxes)
        result = self._estimate(rgb, depth_m, frame.K, initial_masks=masks)
        if not all(result.valid.values()):
            raise TrackingFailure(str(result.failures))
        return result

    def track(self, frame: RGBDFrame) -> SceneResult:
        K, model, coeffs = self.frame_signature
        if (
            not np.array_equal(frame.K, K)
            or frame.distortion_model != model
            or tuple(frame.distortion_coeffs) != coeffs
        ):
            raise ValueError("Camera parameters changed within episode")
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        self._set_frame_time(frame, first=False)
        self.frame_index += 1
        return self._estimate(rgb, depth_m, frame.K)

    def refine_from_pose(self, name, frame, seed, seed_ns):
        """用同段已验证的位姿对一帧做离线 refinement 和深度校验。"""
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        pose = track_objects(
            {name: self.trackers[name]}, rgb, depth_m, frame.K, self.frame_cache,
            seed_poses={name: seed},
        )[name]
        elapsed = abs(int(frame.timestamp_ns) - int(seed_ns)) / 1e9
        try:
            rendered, motion, continuous = self._pose_geometry(name, pose, frame.K, seed, elapsed)
        except TrackingFailure:
            return None
        metrics, verdict = pose_depth_quality(depth_m[PIXEL_GRID], rendered, self.config)
        if verdict != "VALID" or not continuous:
            return None
        return pose, metrics | motion | {"mask_source": 0}

    def _set_frame_time(self, frame: RGBDFrame, *, first: bool) -> None:
        stamp = int(frame.timestamp_ns)
        sequence = int(frame.source_sequence)
        if stamp <= 0 or sequence <= 0:
            raise ValueError("Camera timestamp and source sequence must be positive")
        self.frame_dt_s = 0.0 if first else (stamp - self.timestamp_ns) / 1e9
        self.source_frame_gap = 0 if first else sequence - self.source_sequence
        if not first:
            if self.frame_dt_s < 0:
                raise ValueError("Camera timestamp moved backwards")
            if self.source_frame_gap <= 0:
                raise ValueError("SceneTracker requires a new camera source sequence")
        self.timestamp_ns = stamp
        self.source_sequence = sequence

    def _detect_boxes(self, rgb, names, *, recovery=False, quick=False):
        labels = (self.prompts[name] for name in names) if recovery else self.prompts.values()
        text = ". ".join(labels) + "."
        if recovery:
            candidates = self.segmenter.detect(
                rgb, text, self.config.recovery_box_threshold, self.config.recovery_text_threshold,
                upscale=not quick,
            )
        else:
            candidates = self.segmenter.detect(rgb, text)
        by_prompt: dict[str, list[np.ndarray]] = {}
        for candidate in candidates:
            prompt = candidate["label"].strip().lower().rstrip(". ")
            by_prompt.setdefault(prompt, []).append(candidate["box_xyxy"])
        boxes, failures = {}, {}
        for name in names:
            prompt = self.prompts[name]
            matches = by_prompt.get(prompt, ())
            if not matches:
                failures[name] = f"no detection for {prompt!r}"
            elif recovery or len(matches) == 1:
                boxes[name] = matches[0]
            else:
                failures[name] = f"expected one detection for {prompt!r}, got {len(matches)}"
        return boxes, failures

    def _pose_geometry(self, name, pose, K, reference, dt):
        """检查候选刚体位姿，生成与当前 Mask 无关的几何证据。"""
        if np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
            raise TrackingFailure("FoundationPose returned nonfinite/invalid pose")
        if not np.allclose(pose[3], (0, 0, 0, 1), atol=POSE_HOMOGENEOUS_ATOL, rtol=0):
            raise TrackingFailure(
                "FoundationPose returned invalid homogeneous final row"
            )
        rotation = pose[:3, :3]
        if (
            np.max(np.abs(rotation.T @ rotation - np.eye(3))) > POSE_ROTATION_ATOL
            or abs(np.linalg.det(rotation) - 1) > POSE_ROTATION_ATOL
        ):
            raise TrackingFailure("FoundationPose returned invalid SO(3) rotation")
        # refinement 使用居中网格位姿，其中心必须位于相机前方。
        centered_z = pose[2] @ self.trackers[name].object_from_center[:, 3]
        if centered_z <= 0:
            raise TrackingFailure(
                "FoundationPose centered pose must be in front of the camera"
            )
        rendered = self.renderers[name].render_roi(
            self.directions, pose, K, PIXEL_STRIDE, 2 * PIXEL_STRIDE
        )
        if reference is None:
            motion = {
                "translation_delta_m": 0.0,
                "rotation_delta_rad": 0.0,
                "pose_dt_s": 0.0,
            }
            continuous = True
        else:
            motion, continuous = pose_motion(
                reference, pose, self.trackers[name], dt, self.config
            )
        return rendered, motion, continuous

    def _estimate(self, rgb, depth_m, K, *, initial_masks=None):
        result = SceneResult(
            poses={},
            quality={name: {} for name in self.trackers},
            valid=dict.fromkeys(self.trackers, False),
        )
        sampled = depth_m[PIXEL_GRID]

        def pose_geometry(name, pose, reference=None):
            previous, dt = None, 0.0
            if reference is not None:
                previous, reference_ns = reference
                dt = (self.timestamp_ns - reference_ns) / 1e9
            try:
                return self._pose_geometry(name, pose, K, previous, dt)
            except TrackingFailure as error:
                result.failures[name] = str(error)

        def evaluate(name, mask, geometry, *, coarse_recovery=False, verified_depth=None):
            if mask is not None:
                self.last_mask_sequence[name] = self.source_sequence
            if geometry is None:
                return False
            rendered, motion, continuous = geometry
            if verified_depth is not None:
                metrics, verdict = verified_depth, "VALID"
            elif mask is None:
                metrics, verdict = pose_depth_quality(sampled, rendered, self.config)
            else:
                metrics, verdict = pose_quality(
                    mask[PIXEL_GRID], sampled, rendered, self.config, coarse_recovery=coarse_recovery,
                )
            metrics.update(
                mask_source=int(mask is not None),
                mask_age_frames=self.source_sequence - self.last_mask_sequence[name],
                dt_s=self.frame_dt_s,
                source_frame_gap=self.source_frame_gap,
                **motion,
            )
            result.quality[name] = metrics
            if not continuous:
                result.failures[name] = "implausible pose jump under configured motion limits"
            elif verdict == "VALID":
                result.failures.pop(name, None)
                return True
            else:
                result.failures[name] = (
                    "insufficient geometric evidence" if verdict == "OCCLUDED"
                    else "depth or silhouette disagreement"
                )
            return False

        def accept(name, pose, *, registered=False):
            if name in self.next_retry:
                del self.next_retry[name]
                logger.info("%s: REACQUIRE -> TRACKING", name)
            self.last_pose[name] = pose, self.timestamp_ns
            xy = (pose @ self.trackers[name].object_from_center)[:2, 3]
            if registered:
                self.motion_priors[name] = KalmanXY(xy, self.timestamp_ns)
            else:
                self.motion_priors[name].update(xy, self.timestamp_ns)
            result.poses[name] = pose
            result.valid[name] = True

        def calibrate_offset(name, pose, mask_box, *, tracker_box=None):
            offset = bbox_center_offset(
                pose @ self.trackers[name].object_from_center, K, mask_box, depth_m.shape,
                tracker_box=tracker_box,
            )
            if offset is None:
                self.bbox_offsets.pop(name, None)
            else:
                self.bbox_offsets[name] = offset

        def register(name, mask, *, quick=False):
            tracker = self.trackers[name]
            valid_depth = mask[PIXEL_GRID] & np.isfinite(sampled) & (sampled >= 0.001)
            if np.count_nonzero(valid_depth) < self.config.min_depth_pixels:
                result.failures[name] = "insufficient segmented depth"
                return
            estimator = tracker.estimator
            previous = estimator.pose_last
            try:
                pose = tracker.register(rgb, depth_m, K, mask, quick=quick)
            except TrackingFailure as error:
                result.failures[name] = str(error)
                estimator.pose_last = previous
                return
            if not evaluate(name, mask, pose_geometry(name, pose), coarse_recovery=quick):
                estimator.pose_last = previous
                return
            box = mask_to_xyxy(mask)
            if self.config.xy_prior_gain and result.quality[name]["occluded_fraction"] == 0:
                calibrate_offset(name, pose, box)
            else:
                self.bbox_offsets.pop(name, None)
            self.ostrack.initialize(name, rgb, box)
            accept(name, pose, registered=True)

        if initial_masks is not None:
            for name in self.trackers:
                register(name, initial_masks[name])
            return result

        active = {
            name: tracker for name, tracker in self.trackers.items()
            if name not in self.next_retry
        }
        boxes = {}
        xy_deltas = {}
        previous = {name: tracker.estimator.pose_last for name, tracker in active.items()}
        for name, tracker in active.items():
            box = self.ostrack.track(name, rgb)
            if box is not None:
                boxes[name] = box
            offset = self.bbox_offsets.get(name)
            predicted = (
                self.motion_priors[name].predict(self.timestamp_ns)
                if self.config.kalman_xy_prior else None
            )
            gain = self.config.xy_prior_gain
            if predicted is not None or (box is not None and offset is not None and gain):
                xy_deltas[name] = refinement_xy_delta(
                    self.last_pose[name][0] @ tracker.object_from_center,
                    K, depth_m.shape, box, offset, gain, predicted,
                    min(0.03, 0.25 * tracker.estimator.diameter),
                )
        candidates = (
            track_objects(
                active, rgb, depth_m, K, self.frame_cache,
                xy_deltas=xy_deltas,
            )
            if active else {}
        )
        # 深度异常补分割复用本帧的 CAD 深度和运动检查。
        checks = {
            name: pose_geometry(name, pose, self.last_pose[name])
            for name, pose in candidates.items()
        }
        elapsed = self.frame_index - self.last_sam_frame
        interval = self.config.segmentation_interval_frames
        periodic = elapsed >= interval
        depth_valid = {
            name: evaluate(name, None, geometry)
            for name, geometry in checks.items()
            if not periodic or name not in boxes
        }
        early_segmentation = not periodic and elapsed >= max(2, interval // 2) and any(
            name in boxes and not valid and (geometry := checks[name]) is not None and geometry[2]
            for name, valid in depth_valid.items()
        )
        active_boxes = boxes if periodic or early_segmentation else {}
        masks = self.segmenter.segment_boxes(rgb, active_boxes) if active_boxes else {}
        if active_boxes:
            self.last_sam_frame = self.frame_index

        for name, pose in candidates.items():
            mask = masks.get(name)
            if mask is None and name in depth_valid:
                valid = depth_valid[name]
            else:
                valid = evaluate(name, mask, checks[name])
            if valid:
                accept(name, pose)
                box = boxes.get(name)
                if (
                    mask is not None and box is not None and self.config.xy_prior_gain
                    and result.quality[name]["occluded_fraction"] == 0
                ):
                    calibrate_offset(name, pose, mask_to_xyxy(mask), tracker_box=box)
                continue
            self.trackers[name].estimator.pose_last = previous[name]
            last_valid_ns = self.last_pose[name][1]
            if (
                self.config.recovery_enabled
                and (self.timestamp_ns - last_valid_ns) / 1e9 >= self.config.lost_timeout_s
            ):
                self.next_retry[name] = self.frame_index
                logger.info("%s: TRACKING -> REACQUIRE", name)

        missing = [name for name, valid in result.valid.items() if not valid]
        # 只补紧接可靠观测的一帧。备用位姿从不进入跟踪历史或更新滤波器。
        if missing and 0 < self.frame_dt_s < self.config.lost_timeout_s and all(
            (geometry := checks.get(name)) is not None and geometry[2]
            and geometry[1]["pose_dt_s"] == self.frame_dt_s
            for name in missing
        ):
            fallback = {}
            seeds = {}
            deltas = {}
            for name in missing:
                tracker = active[name]
                rendered, motion, _ = checks[name]
                limit = tracker.estimator.diameter
                depth_metrics, _ = pose_depth_quality(sampled, rendered, self.config)
                if (
                    motion["translation_delta_m"] <= limit
                    and depth_metrics["depth_pixels"] >= max(16, self.config.min_depth_pixels)
                    and depth_metrics["depth_fraction"] >= max(0.75, self.config.min_depth_fraction)
                    and depth_metrics["depth_coverage"] >= self.config.min_depth_coverage
                ):
                    fallback[name] = candidates[name], None, checks[name], depth_metrics
                    continue
                predicted = self.motion_priors[name].predict(self.timestamp_ns)
                if predicted is None or self.config.kalman_xy_prior:
                    break
                seeds[name] = self.last_pose[name][0]
                deltas[name] = refinement_xy_delta(
                    seeds[name] @ tracker.object_from_center,
                    K, depth_m.shape, boxes.get(name), self.bbox_offsets.get(name),
                    self.config.xy_prior_gain, predicted, min(0.03, 0.25 * limit),
                )
            if len(fallback) + len(seeds) == len(missing):
                original_quality = {name: result.quality[name] for name in missing}
                original_failures = {name: result.failures[name] for name in missing}
                if seeds:
                    alternatives = track_objects(
                        {name: active[name] for name in seeds}, rgb, depth_m, K,
                        self.frame_cache, xy_deltas=deltas, seed_poses=seeds,
                    )
                    for name, pose in alternatives.items():
                        geometry = pose_geometry(name, pose, self.last_pose[name])
                        if geometry is not None and geometry[1]["translation_delta_m"] > active[name].estimator.diameter:
                            geometry = None
                        fallback[name] = pose, masks.get(name), geometry, None
                if all(
                    evaluate(name, mask, geometry, verified_depth=verified_depth)
                    or (mask is not None and evaluate(name, None, geometry))
                    for name, (_, mask, geometry, verified_depth) in fallback.items()
                ):
                    for name, (pose, _, _, _) in fallback.items():
                        result.valid[name] = True
                        result.poses[name] = pose
                        result.quality[name]["provisional"] = 1
                else:
                    result.quality.update(original_quality)
                    result.failures.update(original_failures)

        due = [
            name for name in self.trackers
            if name in self.next_retry and self.frame_index >= self.next_retry[name]
        ]
        if due:
            for name in due:
                self.next_retry[name] = self.frame_index + self.config.recovery_retry_interval_frames
            # 旧框可能仍框住手，几何匹配只能验证形状，恢复时需要重新确认物体。
            quick = all(name in active for name in due)
            detection_image, origin = rgb, (0, 0)
            if self.config.recovery_crop and quick and len(due) == 1 and due[0] in boxes:
                detection_image, origin = recovery_image_crop(rgb, boxes[due[0]])
            recovered_boxes, failures = self._detect_boxes(
                detection_image, due, recovery=True, quick=quick,
            )
            result.failures.update(failures)
            if recovered_boxes:
                if origin != (0, 0):
                    recovered_boxes = {name: box + origin * 2 for name, box in recovered_boxes.items()}
                for name, mask in self.segmenter.segment_boxes(
                    rgb, recovered_boxes, reuse_image=bool(active_boxes)
                ).items():
                    register(name, mask, quick=name in active)

        for name, valid in result.valid.items():
            if not valid:
                reason = result.failures.get(name, "awaiting recovery retry")
                state = "REACQUIRE" if name in self.next_retry else "TRACKING"
                result.failures[name] = f"{state}; {reason}"
        if due or missing:
            result.poses = {name: result.poses[name] for name in self.trackers if result.valid[name]}
        return result

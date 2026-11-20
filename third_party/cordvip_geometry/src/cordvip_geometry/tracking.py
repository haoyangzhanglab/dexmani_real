"""结合可见轮廓与实测深度的多物体位姿跟踪。"""

import logging
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .camera import RGBDFrame, RGBDRectifier, camera_rays
from .foundationpose import (
    FoundationPoseTracker,
    TrackingFailure,
    bbox_center_offset,
    track_objects,
)
from .mesh import MeshRaycaster, load_mesh
from .ostrack import OSTrackAdapter, mask_to_xyxy
from .segmentation import SAM2Segmenter, normalize_prompt
from .weights import DEFAULT_MODELS_DIR, GROUNDING_WEIGHTS_DIR, resolve_sam_model

if TYPE_CHECKING:
    from torch import Tensor

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

    segmentation_interval_frames: int = 10  # 活动物体的低频周期校正。
    occlusion_timeout_s: float = 0.5
    max_translation_speed_m_s: float | None = None
    max_rotation_speed_rad_s: float | None = None

    recovery_enabled: bool = True
    recovery_retry_interval_frames: int | None = None
    recovery_box_threshold: float = 0.3
    recovery_text_threshold: float = 0.25

    tracker_2d: str = "sam2_video"
    ostrack_checkpoint: str = "third_party/weights/ostrack/OSTrack_ep0300.pth.tar"
    lost_timeout_s: float = 0.3
    xy_prior_gain: float = 0.2

    def __post_init__(self):
        if self.tracker_2d not in ("sam2_video", "ostrack"):
            raise ValueError("tracker_2d must be 'sam2_video' or 'ostrack'")
        if not isinstance(self.ostrack_checkpoint, str) or not self.ostrack_checkpoint.strip():
            raise ValueError("ostrack_checkpoint must be a nonempty path")
        if self.recovery_retry_interval_frames is None:
            object.__setattr__(
                self, "recovery_retry_interval_frames", 8 if self.tracker_2d == "ostrack" else 15
            )
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
            "occlusion_timeout_s",
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
        if type(self.recovery_enabled) is not bool:
            raise ValueError("recovery_enabled must be bool")
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


def _classify_quality(metrics, config):
    """分别判断观测是否充分和是否矛盾，不把观测缺失当作位姿错误。"""
    if metrics["mask_pixels"] < 0:
        if (
            metrics["depth_support_pixels"] < config.min_depth_pixels
            or metrics["depth_coverage"] < config.min_depth_coverage
        ):
            return "OCCLUDED"
        if metrics["depth_fraction"] < config.min_depth_fraction:
            return "LOST"
        return (
            "VALID"
            if metrics["depth_pixels"] >= config.min_depth_pixels
            else "OCCLUDED"
        )

    mask_supported = (
        metrics["depth_support_pixels"] >= config.min_depth_pixels
        and metrics["mask_depth_coverage"] >= config.min_depth_coverage
    )
    visible_supported = (
        metrics["visible_depth_pixels"] >= config.min_depth_pixels
        and metrics["visible_depth_coverage"] >= config.min_depth_coverage
    )
    if (
        mask_supported
        and (
            metrics["depth_fraction"] < config.min_depth_fraction
            or metrics["supported_recall"] < config.min_recall
        )
    ) or (visible_supported and metrics["supported_precision"] < config.min_precision):
        return "LOST"
    if (
        mask_supported
        and visible_supported
        and metrics["depth_pixels"] >= config.min_depth_pixels
        and metrics["precision"] >= config.min_precision
        and metrics["recall"] >= config.min_recall
    ):
        return "VALID"
    return "OCCLUDED"


def pose_quality(mask, observed, rendered, config):
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
    return metrics, _classify_quality(metrics, config)


def pose_depth_quality(observed, rendered, config):
    """没有当前 mask 时，所有方向的有效深度偏差都计为不匹配。"""
    projected = np.isfinite(rendered) & (rendered > 0)
    support = projected & np.isfinite(observed) & (observed >= 0.001)
    residual = np.abs(observed[support] - rendered[support])
    matched_count = int(np.count_nonzero(residual <= config.depth_tolerance_m))
    support_count = len(residual)
    projected_count = int(projected.sum())
    coverage = support_count / max(1, projected_count)
    fraction = matched_count / max(1, support_count)
    metrics: dict[str, float | int | None] = {
        "mask_pixels": -1,
        "visible_pixels": -1,
        "precision": -1,
        "recall": -1,
        "iou": -1,
        "occluded_fraction": -1,
        "depth_pixels": matched_count,
        "depth_fraction": fraction,
        "depth_coverage": coverage,
        "depth_support_pixels": support_count,
        "visible_depth_pixels": -1,
        "mask_depth_coverage": -1,
        "visible_depth_coverage": -1,
        "supported_precision": -1,
        "supported_recall": -1,
        "median_depth_error_m": float(np.median(residual)) if len(residual) else None,
    }
    return metrics, _classify_quality(metrics, config)


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
        import torch

        self.config = config
        workspace = Path(workspace)
        self.ostrack = (
            OSTrackAdapter(
                workspace / "third_party/OSTrack",
                workspace / Path(config.ostrack_checkpoint).expanduser(),
            )
            if config.tracker_2d == "ostrack"
            else None
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
                load_mesh(path),
                shared_with=shared,
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = MeshRaycaster(tracker.mesh)

        # 预热固定尺寸网络，让 worker 就绪后可直接处理首帧。
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
        self._set_frame_time(frame, first=True)
        if self.ostrack is None:
            self.states = dict.fromkeys(self.trackers, "TRACKING")
            self.occluded_since: dict[str, int] = {}
            # 视频模式保存待确认的 (物体原点位姿, 居中网格位姿)。
            self.pending_pose: dict[str, tuple[np.ndarray, Tensor]] = {}
            self.last_sam_sequence = self.source_sequence
        else:
            self.ostrack.clear()
            self.invalid_since: dict[str, int] = {}
            self.last_good_bbox: dict[str, tuple[float, float, float, float]] = {}
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

        if self.ostrack is None:
            masks = self.segmenter.initialize(rgb, boxes)
            result = self._estimate_sam2_video(rgb, depth_m, frame.K, masks)
        else:
            masks = self.segmenter.segment_boxes(rgb, boxes)
            result = self._estimate_ostrack(self.ostrack, rgb, depth_m, frame.K, initial_masks=masks)
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
        if self.ostrack is not None:
            return self._estimate_ostrack(self.ostrack, rgb, depth_m, frame.K)
        return self._estimate_sam2_video(rgb, depth_m, frame.K)

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

    def _detect_boxes(self, rgb, names, *, recovery=False):
        thresholds: tuple[float, ...]
        if recovery:
            prompts = [self.prompts[name] for name in names]
            thresholds = (self.config.recovery_box_threshold, self.config.recovery_text_threshold)
        else:
            prompts = list(self.prompts.values())
            thresholds = ()
        candidates = self.segmenter.detect(rgb, ". ".join(prompts) + ".", *thresholds)
        by_prompt: dict[str, list[np.ndarray]] = {}
        for candidate in candidates:
            prompt = candidate["label"].strip().lower().rstrip(". ")
            by_prompt.setdefault(prompt, []).append(candidate["box_xyxy"])
        boxes, failures = {}, {}
        for name in names:
            prompt = self.prompts[name]
            matches = by_prompt.get(prompt, ())
            if len(matches) == 1:
                boxes[name] = matches[0]
            else:
                failures[name] = f"expected one detection for {prompt!r}, got {len(matches)}"
        return boxes, failures

    def _redetect_masks(self, rgb, names, result):
        boxes, failures = self._detect_boxes(rgb, names, recovery=True)
        result.failures.update(failures)
        return self.segmenter.segment_boxes(rgb, boxes) if boxes else {}

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

    def _refresh_masks(self, rgb):
        masks = self.segmenter.track(rgb)
        self.last_sam_frame = self.frame_index
        self.last_sam_sequence = self.source_sequence
        return masks

    def _measure_pose(self, result, name, sampled, mask, geometry, last_mask_sequence):
        rendered, motion, continuous = geometry
        if mask is None:
            metrics, verdict = pose_depth_quality(sampled, rendered, self.config)
        else:
            metrics, verdict = pose_quality(mask[PIXEL_GRID], sampled, rendered, self.config)
        metrics.update(
            mask_source=int(mask is not None),
            mask_age_frames=self.source_sequence - last_mask_sequence,
            dt_s=self.frame_dt_s,
            source_frame_gap=self.source_frame_gap,
            **motion,
        )
        result.quality[name] = metrics
        if not continuous:
            result.failures[name] = "implausible pose jump under configured motion limits"
            return "LOST"
        if verdict == "VALID":
            result.failures.pop(name, None)
        else:
            result.failures[name] = (
                "insufficient geometric evidence" if verdict == "OCCLUDED"
                else "depth or silhouette disagreement"
            )
        return verdict

    def _register_pose(self, name, rgb, depth_m, K, mask, sampled):
        if mask is None:
            raise TrackingFailure("missing registration mask")
        valid = mask[PIXEL_GRID] & np.isfinite(sampled) & (sampled >= 0.001)
        if np.count_nonzero(valid) < self.config.min_depth_pixels:
            raise TrackingFailure("insufficient segmented depth")
        return self.trackers[name].register(rgb, depth_m, K, mask)

    def _estimate_ostrack(self, ostrack: OSTrackAdapter, rgb, depth_m, K, *, initial_masks=None):
        result = SceneResult(
            poses={},
            quality={name: {} for name in self.trackers},
            valid=dict.fromkeys(self.trackers, False),
        )
        sampled = depth_m[PIXEL_GRID]

        def evaluate(name, pose, mask, reference=None):
            if mask is not None:
                self.last_mask_sequence[name] = self.source_sequence
            previous, dt = None, 0.0
            if reference is not None:
                previous, reference_ns = reference
                dt = (self.timestamp_ns - reference_ns) / 1e9
            try:
                geometry = self._pose_geometry(name, pose, K, previous, dt)
            except TrackingFailure as error:
                result.failures[name] = str(error)
                return False
            return self._measure_pose(
                result, name, sampled, mask, geometry, self.last_mask_sequence[name],
            ) == "VALID"

        def accept(name, pose):
            if name in self.next_retry:
                del self.next_retry[name]
                logger.info("%s: REACQUIRE -> TRACKING", name)
            self.invalid_since.pop(name, None)
            self.last_pose[name] = pose, self.timestamp_ns
            result.poses[name] = pose
            result.valid[name] = True

        def register(name, mask):
            tracker = self.trackers[name]
            previous = tracker.estimator.pose_last
            try:
                pose = self._register_pose(name, rgb, depth_m, K, mask, sampled)
            except TrackingFailure as error:
                result.failures[name] = str(error)
                tracker.estimator.pose_last = previous
                return False
            if not evaluate(name, pose, mask):
                tracker.estimator.pose_last = previous
                return False
            # 通过 segmented depth 门槛的 Mask 一定能生成非空包围框。
            box = mask_to_xyxy(mask)
            self.last_good_bbox[name] = box
            if self.config.xy_prior_gain:
                offset = bbox_center_offset(
                    pose @ tracker.object_from_center, K, box, depth_m.shape
                )
                if offset is None:
                    self.bbox_offsets.pop(name, None)
                else:
                    self.bbox_offsets[name] = offset
            ostrack.initialize(name, rgb, box)
            accept(name, pose)
            return True

        if initial_masks is not None:
            for name in self.trackers:
                register(name, initial_masks.get(name))
            return result

        active = {
            name: tracker for name, tracker in self.trackers.items()
            if name not in self.next_retry
        }
        due = [
            name for name in self.trackers
            if self.config.recovery_enabled
            and name in self.next_retry
            and self.frame_index >= self.next_retry[name]
        ]
        boxes = {}
        priors = {}
        previous = {name: tracker.estimator.pose_last for name, tracker in active.items()}
        for name, tracker in active.items():
            box = ostrack.track(name, rgb)
            if box is None:
                continue
            boxes[name] = box
            if name in self.bbox_offsets:
                priors[name] = (
                    self.last_pose[name][0] @ tracker.object_from_center,
                    box,
                    self.bbox_offsets[name],
                )
        candidates = (
            track_objects(
                active, rgb, depth_m, K, self.frame_cache,
                bbox_priors=priors, xy_prior_gain=self.config.xy_prior_gain,
            )
            if active else {}
        )
        segment_boxes = {name: self.last_good_bbox[name] for name in due}
        if (
            active
            and self.frame_index - self.last_sam_frame >= self.config.segmentation_interval_frames
        ):
            segment_boxes.update(boxes)
            self.last_sam_frame = self.frame_index
        masks = self.segmenter.segment_boxes(rgb, segment_boxes) if segment_boxes else {}

        for name, pose in candidates.items():
            mask = masks.get(name)
            if evaluate(name, pose, mask, self.last_pose[name]):
                accept(name, pose)
                box = boxes.get(name)
                if box is not None and (
                    mask is not None
                    or bbox_center_offset(
                        pose @ self.trackers[name].object_from_center, K, box, depth_m.shape
                    ) is not None
                ):
                    self.last_good_bbox[name] = box
                continue
            self.trackers[name].estimator.pose_last = previous[name]
            since = self.invalid_since.setdefault(name, self.timestamp_ns)
            if (self.timestamp_ns - since) / 1e9 >= self.config.lost_timeout_s:
                del self.invalid_since[name]
                self.next_retry[name] = self.frame_index + 1
                logger.info("%s: TRACKING -> REACQUIRE", name)

        retry_dino = []
        for name in due:
            self.next_retry[name] = self.frame_index + cast(int, self.config.recovery_retry_interval_frames)
            if not register(name, masks.get(name)):
                retry_dino.append(name)
        if retry_dino:
            for name, mask in self._redetect_masks(rgb, retry_dino, result).items():
                register(name, mask)

        for name, valid in result.valid.items():
            if not valid:
                reason = result.failures.get(name, "awaiting recovery retry")
                state = "REACQUIRE" if name in self.next_retry else "TRACKING"
                result.failures[name] = f"{state}; {reason}"
        if due:
            result.poses = {name: result.poses[name] for name in self.trackers if result.valid[name]}
        return result

    def _estimate_sam2_video(self, rgb, depth_m, K, masks=None):
        first = masks is not None
        # 后端替换 pose_last 而不原地修改旧张量；这些引用始终指向最后可靠位姿。
        previous = {
            name: tracker.estimator.pose_last for name, tracker in self.trackers.items()
        }
        result = SceneResult(
            poses={},
            quality={name: {} for name in self.trackers},
            valid=dict.fromkeys(self.trackers, False),
        )
        confirming = self.pending_pose
        # 本帧新丢失（包括遮挡超时）的对象，从下一处理帧才开始检测恢复。
        due = [
            name
            for name in self.trackers
            if self.config.recovery_enabled
            and self.states[name] == "LOST"
            and name not in confirming
            and self.frame_index >= self.next_retry[name]
        ]
        candidates: dict[str, np.ndarray] = {}
        verdicts = dict.fromkeys(self.trackers, "LOST")
        pose_checks = {}
        sampled = depth_m[PIXEL_GRID]

        def reject(name, error):
            verdicts[name] = "LOST"
            result.failures[name] = str(error)

        expired = {
            name
            for name, since in self.occluded_since.items()
            if (self.timestamp_ns - since) / 1e9 >= self.config.occlusion_timeout_s
        }
        for name in expired:
            reject(name, "occlusion timeout")

        active = {
            name: tracker
            for name, tracker in self.trackers.items()
            if name not in expired
            and (self.states[name] != "LOST" or name in confirming)
        }
        if masks is None and (
            due
            or confirming
            or (
                active
                and self.frame_index - self.last_sam_frame
                >= self.config.segmentation_interval_frames
            )
        ):
            masks = self._refresh_masks(rgb)

        def evaluate(name, pose, mask) -> None:
            if name not in pose_checks:
                reference, dt = None, 0.0
                if name in confirming:
                    reference = self.pending_pose[name][0]
                    dt = self.frame_dt_s
                elif name in self.last_pose and name not in due:
                    reference, reference_ns = self.last_pose[name]
                    dt = (self.timestamp_ns - reference_ns) / 1e9
                # Mask 更新后仅重算可见证据，复用同一候选的位姿检查和渲染。
                pose_checks[name] = self._pose_geometry(name, pose, K, reference, dt)
            verdicts[name] = self._measure_pose(
                result, name, sampled, mask, pose_checks[name], self.last_sam_sequence,
            )

        def check_candidates():
            for name, pose in list(candidates.items()):
                try:
                    evaluate(name, pose, None if masks is None else masks.get(name))
                except TrackingFailure as error:
                    reject(name, error)
                    del candidates[name]

        def register(name, mask):
            try:
                pose = self._register_pose(name, rgb, depth_m, K, mask, sampled)
                evaluate(name, pose, mask)
                if verdicts[name] != "VALID":
                    return False
            except TrackingFailure as error:
                reject(name, error)
                return False
            candidates[name] = pose
            return True

        if first:
            for name in active:
                register(name, masks.get(name))
        elif active:
            for name, tracker in active.items():
                if name in confirming:
                    tracker.estimator.pose_last = self.pending_pose[name][1]
            candidates = track_objects(active, rgb, depth_m, K, self.frame_cache)
            check_candidates()
            if masks is None and any(
                continuous and verdicts[name] != "VALID"
                for name, (_, _, continuous) in pose_checks.items()
            ):
                masks = self._refresh_masks(rgb)
                # 用当前轮廓区分前景遮挡与位姿不一致。
                check_candidates()

        if due:
            for name in due:
                self.next_retry[name] = (
                    self.frame_index + cast(int, self.config.recovery_retry_interval_frames)
                )
            corrections = {
                name: mask
                for name, mask in self._redetect_masks(rgb, due, result).items()
                if register(name, mask)
            }
            if corrections:
                masks = self.segmenter.correct_masks(corrections)
                # 校正可能改变任意对象的可见边界；复用各候选的 CAD 深度复查。
                check_candidates()

        # Mask 合并后统一提交或恢复内部位姿；只有确认过的当前观测进入 poses。
        for name, tracker in self.trackers.items():
            state = self.states[name]
            verdict = verdicts[name]
            if name in due and verdict == "VALID":
                self.pending_pose[name] = (
                    candidates[name],
                    tracker.estimator.pose_last,
                )
                tracker.estimator.pose_last = previous[name]
                result.failures[name] = "LOST; awaiting second recovery observation"
                continue
            if verdict == "VALID":
                self.states[name] = "TRACKING"
                self.next_retry.pop(name, None)
                self.last_pose[name] = candidates[name], self.timestamp_ns
                result.poses[name] = candidates[name]
                result.valid[name] = True
            else:
                tracker.estimator.pose_last = previous[name]
                if verdict == "OCCLUDED" and state != "LOST" and not first:
                    self.states[name] = "OCCLUDED"
                    self.occluded_since.setdefault(name, self.timestamp_ns)
                else:
                    self.states[name] = "LOST"
                    if state != "LOST":
                        self.next_retry[name] = self.frame_index + 1
                reason = result.failures.get(name, "awaiting recovery retry")
                result.failures[name] = f"{self.states[name]}; {reason}"
            if self.states[name] != "OCCLUDED":
                self.occluded_since.pop(name, None)
            self.pending_pose.pop(name, None)
            if state != self.states[name]:
                logger.info("%s: %s -> %s", name, state, self.states[name])
        return result

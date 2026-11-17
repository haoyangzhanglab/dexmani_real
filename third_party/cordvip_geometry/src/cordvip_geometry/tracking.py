"""结合可见轮廓与实测深度的多物体位姿跟踪。"""

import logging
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .camera import RGBDFrame, RGBDRectifier, camera_rays
from .foundationpose import (
    FoundationPoseTracker,
    TrackingFailure,
    track_objects,
)
from .mesh import MeshRaycaster, load_mesh
from .segmentation import SAM2Segmenter, normalize_prompt

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)
PIXEL_STRIDE = 2
# 容纳 float32 迭代误差，拒绝明显非刚体候选。
POSE_ROTATION_ATOL = 1e-3
POSE_HOMOGENEOUS_ATOL = 1e-6


@dataclass(frozen=True)
class TrackingConfig:
    depth_tolerance_m: float = 0.015
    min_depth_pixels: int = 30
    min_depth_fraction: float = 0.5
    min_depth_coverage: float = 0.5
    min_precision: float = 0.5
    min_recall: float = 0.5

    segmentation_interval_frames: int = 2
    occlusion_timeout_s: float = 0.5
    max_translation_speed_m_s: float | None = None
    max_rotation_speed_rad_s: float | None = None

    recovery_enabled: bool = True
    recovery_retry_interval_frames: int = 15
    recovery_box_threshold: float = 0.3
    recovery_text_threshold: float = 0.25

    def __post_init__(self):
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
        and metrics["depth_fraction"] >= config.min_depth_fraction
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
    """在居中网格坐标中比较等价位姿，报告同一候选的平移和转角。"""
    previous = previous @ tracker.object_from_center
    equivalent = (current @ tracker.object_from_center)[None] @ tracker.symmetry_tfs
    translations = np.linalg.norm(equivalent[:, :3, 3] - previous[:3, 3], axis=1)
    rotations = np.arccos(
        np.clip(
            (np.einsum("ij,nij->n", previous[:3, :3], equivalent[:, :3, :3]) - 1) / 2,
            -1,
            1,
        )
    )
    diameter = tracker.estimator.diameter
    allowed = np.ones(len(equivalent), bool)
    # 没有实测上限时只报告变化量；零 dt 也不能据此推断无限速度。
    if dt > 0:
        if config.max_translation_speed_m_s is not None:
            allowed &= (
                translations
                <= config.max_translation_speed_m_s * dt + config.depth_tolerance_m
            )
        if config.max_rotation_speed_rad_s is not None:
            angular_slack = 2 * np.arcsin(min(1, config.depth_tolerance_m / diameter))
            allowed &= rotations <= config.max_rotation_speed_rad_s * dt + angular_slack
    indices = np.flatnonzero(allowed)
    if not len(indices):
        indices = np.arange(len(equivalent))
    index = indices[np.argmin(rotations[indices])]
    return {
        "translation_delta_m": float(translations[index]),
        "rotation_delta_rad": float(rotations[index]),
        "pose_dt_s": dt,
    }, bool(allowed.any())


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
        self.states = dict.fromkeys(self.trackers, "TRACKING")
        self.next_retry = dict.fromkeys(self.trackers, 0)
        self.occluded_since: dict[str, int] = {}
        # Pending 保存 (物体原点位姿, 居中网格内部位姿)，不提交到 pose_last。
        self.pending_pose: dict[str, tuple[np.ndarray, Tensor]] = {}
        self.last_pose: dict[str, np.ndarray] = {}
        self.last_pose_ns: dict[str, int] = {}
        self.frame_index = 0
        self.last_sam_frame = 0
        self._set_frame_time(frame, first=True)
        self.last_sam_sequence = self.source_sequence
        self.frame_signature = (
            frame.depth_m.shape,
            frame.K.copy(),
            frame.distortion_model,
            np.asarray(frame.distortion_coeffs).copy(),
        )
        self.frame_cache: dict[str, Any] = {}
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

        candidates = self.segmenter.detect(
            rgb, ". ".join(self.prompts.values()) + "."
        )
        boxes = {}
        for name in self.trackers:
            try:
                boxes[name] = self._match_box(candidates, self.prompts[name])
            except TrackingFailure as error:
                raise TrackingFailure(f"{name}: {error}") from error

        masks = self.segmenter.initialize(rgb, boxes)
        result = self._estimate(rgb, depth_m, frame.K, masks, first=True)
        if not all(result.valid.values()):
            raise TrackingFailure(str(result.failures))
        return result

    def track(self, frame: RGBDFrame) -> SceneResult:
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
        self._set_frame_time(frame, first=False)
        self.frame_index += 1
        return self._estimate(rgb, depth_m, frame.K, None, first=False)

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

    def _pose_geometry(self, name, pose, reference, dt):
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
        rendered = (
            self.renderers[name]
            .render(self.directions.reshape(-1, 3), pose)
            .reshape(self.directions.shape[:2])
        )
        if reference is None:
            motion = dict(
                translation_delta_m=0.0, rotation_delta_rad=0.0, pose_dt_s=0.0
            )
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

    def _estimate(self, rgb, depth_m, K, masks, *, first):
        # 后端替换 pose_last 而不原地修改旧张量；这些引用始终指向最后可靠位姿。
        previous = {
            name: tracker.estimator.pose_last for name, tracker in self.trackers.items()
        }
        result = SceneResult(
            poses={},
            quality={name: {} for name in self.trackers},
            valid=dict.fromkeys(self.trackers, False),
        )
        confirming = set(self.pending_pose) if self.config.recovery_enabled else set()
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
        registered = set()
        pose_checks = {}
        sampled = depth_m[::PIXEL_STRIDE, ::PIXEL_STRIDE]

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

        def evaluate(name, pose, mask):
            if name not in pose_checks:
                reference, dt = None, 0.0
                if name in confirming:
                    reference = self.pending_pose[name][0]
                    dt = self.frame_dt_s
                elif name in self.last_pose and name not in registered:
                    reference = self.last_pose[name]
                    dt = (self.timestamp_ns - self.last_pose_ns[name]) / 1e9
                # Mask 更新后仅重算可见证据，复用同一候选的位姿检查和渲染。
                pose_checks[name] = self._pose_geometry(name, pose, reference, dt)
            rendered, motion, continuous = pose_checks[name]
            if mask is None:
                metrics, verdict = pose_depth_quality(sampled, rendered, self.config)
            else:
                metrics, verdict = pose_quality(
                    mask[::PIXEL_STRIDE, ::PIXEL_STRIDE], sampled, rendered, self.config
                )
            metrics.update(
                mask_source=int(mask is not None),
                mask_age_frames=self.source_sequence - self.last_sam_sequence,
                dt_s=self.frame_dt_s,
                source_frame_gap=self.source_frame_gap,
                **motion,
            )
            result.quality[name] = metrics
            if not continuous:
                verdict = "LOST"
                result.failures[name] = (
                    "implausible pose jump under configured motion limits"
                )
            elif verdict == "OCCLUDED":
                result.failures[name] = "insufficient geometric evidence"
            elif verdict == "LOST":
                result.failures[name] = "depth or silhouette disagreement"
            else:
                result.failures.pop(name, None)
            verdicts[name] = verdict
            return verdict

        def check_candidates():
            for name, pose in list(candidates.items()):
                try:
                    evaluate(name, pose, None if masks is None else masks.get(name))
                except TrackingFailure as error:
                    reject(name, error)
                    del candidates[name]

        def register(name, mask):
            try:
                if mask is None:
                    raise TrackingFailure("missing registration mask")
                valid = (
                    mask[::PIXEL_STRIDE, ::PIXEL_STRIDE]
                    & np.isfinite(sampled)
                    & (sampled >= 0.001)
                )
                if np.count_nonzero(valid) < self.config.min_depth_pixels:
                    raise TrackingFailure("insufficient segmented depth")
                pose = self.trackers[name].register(rgb, depth_m, K, mask)
                registered.add(name)
                if evaluate(name, pose, mask) != "VALID":
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
            if masks is None and any(verdicts[name] != "VALID" for name in candidates):
                masks = self._refresh_masks(rgb)
                # 更近的深度先触发当前 Mask，再决定它是否属于前景遮挡。
                check_candidates()

        if due:
            for name in due:
                self.next_retry[name] = (
                    self.frame_index + self.config.recovery_retry_interval_frames
                )
            detections = self.segmenter.detect(
                rgb,
                ". ".join(self.prompts[name] for name in due) + ".",
                self.config.recovery_box_threshold,
                self.config.recovery_text_threshold,
            )
            boxes = {}
            for name in due:
                try:
                    boxes[name] = self._match_box(detections, self.prompts[name])
                except TrackingFailure as error:
                    reject(name, error)
            recovered_masks = (
                self.segmenter.segment_boxes(rgb, boxes) if boxes else {}
            )
            corrections = {
                name: mask
                for name, mask in recovered_masks.items()
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
            if state == "LOST" and name in registered and verdict == "VALID":
                self.pending_pose[name] = (
                    candidates[name],
                    tracker.estimator.pose_last,
                )
                tracker.estimator.pose_last = previous[name]
                result.failures[name] = "LOST; awaiting second recovery observation"
                continue
            if verdict == "VALID":
                self.states[name] = "TRACKING"
                self.last_pose[name] = candidates[name]
                self.last_pose_ns[name] = self.timestamp_ns
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

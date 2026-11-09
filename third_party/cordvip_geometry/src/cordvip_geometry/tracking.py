"""同行分割、实测点云与可选模型位姿。"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .foundationpose import MIN_DEPTH_PIXELS, PoseTracker, PoseRegistrationError, prepare_tracking_frame
from .geometry import (
    RectificationMaps,
    transform_points,
    validate_pose,
)
from .model_depth import ModelDepthRenderer, camera_rays


class SceneInitializationError(ValueError):
    """当前场景无法初始化；新一轮可重新尝试，模型和协议错误不属于此类。"""


def validate_object_name(name):
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\0" in name
        or Path(name).name != name
    ):
        raise ValueError("Object names must be nonempty single directory names")


def _validate_tracking_parameters(
    *, has_models, pixel_stride, depth_tolerance_m, min_matched_pixels, min_matched_fraction,
    max_reregistrations,
):
    integer_parameters = [("pixel_stride", pixel_stride, 1)]
    if has_models:
        integer_parameters.extend((
            ("min_matched_pixels", min_matched_pixels, 1),
            ("max_reregistrations", max_reregistrations, 0),
        ))
    for name, value, minimum in integer_parameters:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if has_models:
        for name, value in (
            ("depth_tolerance_m", depth_tolerance_m),
            ("min_matched_fraction", min_matched_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if min_matched_fraction > 1:
            raise ValueError("min_matched_fraction must be in (0,1]")


@dataclass
class ObjectCloud:
    """相机坐标的实测点；模型残差是实测 Z 减模型 Z，无表面处为 NaN。"""

    semantic_mask: np.ndarray
    observed_mask: np.ndarray
    observed_points_camera: np.ndarray
    observed_colors: np.ndarray
    observed_pixel_indices: np.ndarray
    model_supported_mask: np.ndarray | None = None
    observed_model_depth_residual_m: np.ndarray | None = None
    model_supported_point_count: int = 0
    model_supported_fraction: float = 0.0


def extract_scene_clouds(rgb, depth_m, directions, masks, *, pixel_stride=2):
    sample = (slice(None, None, pixel_stride), slice(None, None, pixel_stride))
    sampled_depth, sampled_rgb = depth_m[sample], rgb[sample]
    objects = {}
    for name, semantic in masks.items():
        selected = semantic[sample] & (sampled_depth > 0)
        v, u = np.nonzero(selected)
        pixels = np.column_stack((v * pixel_stride, u * pixel_stride)).astype(np.int32)
        z = sampled_depth[selected]
        points = (directions[selected] * z[:, None]).astype(np.float32)
        observed_mask = np.zeros(depth_m.shape, dtype=bool)
        observed_mask[sample] = selected
        objects[name] = ObjectCloud(
            semantic_mask=semantic,
            observed_mask=observed_mask,
            observed_points_camera=points,
            observed_colors=sampled_rgb[selected],
            observed_pixel_indices=pixels,
        )
    return objects


def add_model_depth_support(measured, model_z_by_object, *, depth_tolerance_m=0.015):
    for name, model_z in model_z_by_object.items():
        cloud = measured[name]
        comparable = model_z > 0
        residual = np.full(len(model_z), np.nan, dtype=np.float32)
        residual[comparable] = (
            cloud.observed_points_camera[comparable, 2] - model_z[comparable]
        )
        supported = comparable & (np.abs(residual) <= depth_tolerance_m)
        pixels = cloud.observed_pixel_indices[supported]
        cloud.model_supported_mask = np.zeros(cloud.observed_mask.shape, dtype=bool)
        cloud.model_supported_mask[pixels[:, 0], pixels[:, 1]] = True
        cloud.observed_model_depth_residual_m = residual
        cloud.model_supported_point_count = len(pixels)
        cloud.model_supported_fraction = (
            len(pixels) / len(model_z) if len(model_z) else 0.0
        )


class SceneTracker:
    def __init__(
        self,
        segmenter,
        assets,
        *,
        foundationpose_root,
        debug_dir,
        pixel_stride=2,
        depth_tolerance_m=0.015,
        min_matched_pixels=30,
        min_matched_fraction=0.15,
        initial_selections=None,
        max_reregistrations=0,
    ):
        self.segmenter = segmenter
        self.assets = assets
        self.initial_selections = initial_selections
        self.pixel_stride = pixel_stride
        self.depth_tolerance_m = depth_tolerance_m
        self.min_matched_pixels = min_matched_pixels
        self.min_matched_fraction = min_matched_fraction
        self.trackers, self.renderers = {}, {}
        self.lost_objects = {}
        self.max_reregistrations = max_reregistrations
        self.registration_attempts = {}
        self.directions = self.maps = None
        self._camera: tuple[np.ndarray, tuple[int, int], str, tuple[float, ...]]
        shared = None
        for name, (mesh, _, _) in assets.items():
            tracker = PoseTracker(
                foundationpose_root, mesh, Path(debug_dir) / name, shared_with=shared
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = ModelDepthRenderer(
                tracker.mesh, pixel_stride=pixel_stride
            )

    @classmethod
    @contextmanager
    def open(
        cls,
        assets,
        *,
        initial_selections=None,
        workspace=None,
        models_dir=None,
        sam_python=None,
        sam_variant=None,
        sam_checkpoint=None,
        sam_config=None,
        log_dir=None,
        pixel_stride=2,
        depth_tolerance_m=0.015,
        min_matched_pixels=30,
        min_matched_fraction=0.15,
        max_reregistrations=0,
    ):
        """一次加载资产与模型，退出时释放 SAM 子进程。"""
        _validate_tracking_parameters(
            has_models=any(path is not None for path in assets.values()),
            pixel_stride=pixel_stride, depth_tolerance_m=depth_tolerance_m,
            min_matched_pixels=min_matched_pixels, min_matched_fraction=min_matched_fraction,
            max_reregistrations=max_reregistrations,
        )
        from .assets import load_asset
        from .models import DEFAULT_MODELS_DIR, resolve_sam_model
        from .sam2 import SAM2Client

        for name in assets:
            validate_object_name(name)
        if initial_selections is not None:
            for name in initial_selections:
                validate_object_name(name)
        workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        models_dir = (
            Path(models_dir or workspace / DEFAULT_MODELS_DIR).expanduser().resolve()
        )
        checkpoint, config = resolve_sam_model(
            models_dir, sam_variant, sam_checkpoint, sam_config
        )
        loaded = {}
        for name, path in assets.items():
            if path is not None:
                path = Path(path).expanduser()
                mesh, points, normals, _ = load_asset(
                    path if path.is_absolute() else workspace / path
                )
                loaded[name] = mesh, points, normals
        if initial_selections is not None:
            initial_selections = {
                name: workspace / Path(path).expanduser()
                for name, path in initial_selections.items()
            }
        with TemporaryDirectory(prefix="cordvip_") as temporary:
            debug_dir = (
                Path(log_dir).expanduser().resolve() if log_dir else Path(temporary)
            )
            with SAM2Client(
                sam_python or sys.executable,
                workspace,
                models_dir,
                checkpoint=checkpoint,
                config=config,
                log_path=debug_dir / "sam2_worker.log" if log_dir else None,
            ) as segmenter:
                tracker = cls(
                    segmenter,
                    loaded,
                    foundationpose_root=workspace / "third_party/FoundationPose",
                    debug_dir=debug_dir / "foundationpose",
                    initial_selections=initial_selections,
                    pixel_stride=pixel_stride,
                    depth_tolerance_m=depth_tolerance_m,
                    min_matched_pixels=min_matched_pixels,
                    min_matched_fraction=min_matched_fraction,
                    max_reregistrations=max_reregistrations,
                )
                try:
                    yield tracker
                finally:
                    tracker.trackers.clear()
                    tracker.renderers.clear()

    def reset(self):
        """重置场景状态，保留模型权重和固定资产。"""
        self.segmenter.reset()
        self.maps = self.directions = None
        self.lost_objects.clear()
        self.registration_attempts.clear()
        for tracker in self.trackers.values():
            tracker.estimator.pose_last = None

    def detect_masks(self, frame, prompts):
        candidates = self.segmenter.predict(frame.rgb, ". ".join(prompts.values()) + ".")
        masks = {}
        for name, prompt in prompts.items():
            matches = [item for item in candidates if item["label"].strip().lower().rstrip(".") == prompt.strip().lower().rstrip(".")]
            if len(matches) != 1:
                raise SceneInitializationError(f"{name}: 首帧候选数 {len(matches)}，无法无歧义初始化")
            masks[name] = matches[0]["mask"]
        return masks

    def step(self, frame, *, initial_masks=None):
        """输入原 color 网格，按调用顺序处理每一帧。"""
        maps = self.maps
        first = maps is None
        if first:
            if initial_masks is not None and self.initial_selections is not None:
                raise ValueError("Choose initial_masks or initial_selections, not both")
            if self.initial_selections is not None:
                from .io import load_initial_selection

                initial_masks = {
                    name: load_initial_selection(path, frame)
                    for name, path in self.initial_selections.items()
                }
            if not isinstance(initial_masks, dict) or not initial_masks:
                raise ValueError(
                    "The first frame requires initial_masks or initial_selections"
                )
            if self.initial_selections is None:
                for name in initial_masks:
                    validate_object_name(name)
        elif initial_masks is not None:
            raise ValueError("Call reset() before initializing another scene")
        if not first:
            K, shape, model, distortion = self._camera
            if (
                frame.depth_m.shape != shape
                or not np.array_equal(frame.K, K)
                or frame.distortion_coeffs != distortion
                or frame.distortion_model.removeprefix("distortion.") != model
            ):
                raise ValueError(
                    "A tracker requires one fixed camera calibration and image grid"
                )
        if maps is None:
            maps = RectificationMaps(
                frame.depth_m.shape, frame.K, frame.distortion_coeffs
            )
        rgb, depth_m = maps.apply(frame.rgb, frame.depth_m)
        K = frame.K
        if first:
            masks = {}
            for name, mask in initial_masks.items():
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != frame.depth_m.shape:
                    raise ValueError(
                        f"Initial mask {name} must use the original bool color grid"
                    )
                mask = maps.apply_mask(mask)
                masks[name] = mask
            if not self.assets.keys() <= masks.keys():
                raise ValueError("Every asset requires a named initial mask")
            # 排除无明确归属的重叠像素，在 IPC 前识别场景初始化失败。
            unique = np.stack(list(masks.values())).sum(axis=0) == 1
            masks = {name: mask & unique for name, mask in masks.items()}
            for name, mask in masks.items():
                if not mask.any():
                    raise SceneInitializationError(f"{name}: initial mask empty after rectification/overlap exclusion")
            masks = self.segmenter.initialize(rgb, masks)
            self.maps = maps
            self._camera = (
                K.copy(),
                frame.depth_m.shape,
                frame.distortion_model.removeprefix("distortion."),
                frame.distortion_coeffs,
            )
            self.directions = camera_rays(depth_m.shape, K, self.pixel_stride)
        else:
            masks = self.segmenter.track(rgb)
        measured = extract_scene_clouds(
            rgb, depth_m, self.directions, masks, pixel_stride=self.pixel_stride
        )
        poses, model_z_by_object = {}, {}
        prepared_frame = None
        for name, tracker in self.trackers.items():
            recovering = name in self.lost_objects
            if recovering and self.registration_attempts.get(name, 0) >= self.max_reregistrations:
                continue
            cloud = measured[name]
            mask = masks[name]
            count = len(cloud.observed_points_camera)
            fp_count = np.count_nonzero(mask & (depth_m >= 0.001))
            if count < self.min_matched_pixels or fp_count < MIN_DEPTH_PIXELS:
                if first:
                    raise SceneInitializationError(
                        f"{name}: pose initialization has {count} independent depth points "
                        f"after segmentation/stride and {fp_count} FoundationPose depth pixels; "
                        f"requires {self.min_matched_pixels} and {MIN_DEPTH_PIXELS}, respectively. "
                        "Select a supported instance and reset the scene."
                    )
                self.lost_objects[name] = (
                    "Insufficient independently segmented depth; pose lost"
                )
                continue
            if not first and not recovering and prepared_frame is None:
                prepared_frame = prepare_tracking_frame(rgb, depth_m, K)
            if recovering:
                self.registration_attempts[name] = self.registration_attempts.get(name, 0) + 1
            try:
                poses[name] = (
                    tracker.register(rgb, depth_m, K, mask)
                    if first or recovering
                    else tracker.track(K, mask, prepared_frame=prepared_frame)
                )
            except PoseRegistrationError as error:
                self.lost_objects[name] = str(error)
                continue
            model_z_by_object[name] = self.renderers[name].render(
                cloud.observed_pixel_indices,
                poses[name],
                directions=self.directions,
            )
        add_model_depth_support(
            measured, model_z_by_object, depth_tolerance_m=self.depth_tolerance_m
        )
        objects = {}
        for name, cloud in measured.items():
            pose = poses.get(name)
            supported = cloud.model_supported_point_count
            fraction = cloud.model_supported_fraction
            if pose is not None and (
                supported < self.min_matched_pixels
                or fraction < self.min_matched_fraction
            ):
                self.lost_objects[name] = "Semantic depth/model disagreement; pose lost"
                pose = None
            pose_valid = pose is not None
            if pose_valid:
                self.lost_objects.pop(name, None)
                self.registration_attempts.pop(name, None)
            objects[name] = {
                "cloud": cloud,
                "pose": pose,
                "pose_valid": pose_valid,
                "state": (
                    "TRACKING"
                    if pose_valid
                    else "POSE_LOST"
                    if name in self.trackers
                    else "OBSERVED_ONLY"
                ),
                "error": self.lost_objects.get(name),
                "candidate_pose": poses.get(name),
                "observed_valid": bool(len(cloud.observed_points_camera)),
            }
        return {
            "frame": frame,
            "rgb": rgb,
            "objects": objects,
            "operation": "initialize" if first else "track",
        }

    def model_points(self, result, name, *, T_target_camera=None):
        """按需由有效位姿重建固定模型点，默认位于 camera frame。"""
        item = result["objects"][name]
        if not item["pose_valid"]:
            return np.empty((0, 3), dtype=np.float32)
        pose = item["pose"]
        if T_target_camera is not None:
            pose = validate_pose(T_target_camera) @ pose
        return transform_points(self.assets[name][1], pose)

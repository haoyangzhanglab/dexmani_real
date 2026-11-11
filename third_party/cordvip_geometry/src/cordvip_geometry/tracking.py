"""语义分割与深度门控的多物体位姿跟踪。"""

import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .foundationpose import MIN_DEPTH_PIXELS, PoseTracker, TrackingFailure, prepare_tracking_frame
from .geometry import RectificationMaps, validate_intrinsics, validate_distortion
from .model_depth import ModelDepthRenderer, camera_rays


PIXEL_STRIDE = 2
DEPTH_TOLERANCE_M = 0.015
MIN_MATCHED_PIXELS = 30
MIN_MATCHED_FRACTION = 0.15


class SceneTracker:
    def __init__(
        self,
        segmenter,
        assets,
        *,
        foundationpose_root,
        debug_dir,
    ):
        self.segmenter = segmenter
        self.trackers, self.renderers = {}, {}
        self.directions = self.maps = None
        shared = None
        for name, mesh in assets.items():
            tracker = PoseTracker(
                foundationpose_root, mesh, Path(debug_dir) / name, shared_with=shared
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = ModelDepthRenderer(
                tracker.mesh, pixel_stride=PIXEL_STRIDE
            )

    @classmethod
    @contextmanager
    def open(cls, assets, *, workspace):
        """一次加载资产与模型，退出时释放 SAM 子进程。"""
        from .assets import load_asset
        from .models import DEFAULT_MODELS_DIR, resolve_sam_model
        from .sam2 import SAM2Client

        if not assets or any(path is None for path in assets.values()):
            raise ValueError("Every task object requires a Mesh asset")
        workspace = Path(workspace).expanduser().resolve()
        models_dir = workspace / DEFAULT_MODELS_DIR
        checkpoint, config = resolve_sam_model(models_dir)
        loaded = {}
        for name, path in assets.items():
            path = Path(path).expanduser()
            mesh, _ = load_asset(path if path.is_absolute() else workspace / path)
            loaded[name] = mesh
        with TemporaryDirectory(prefix="cordvip_") as temporary:
            debug_dir = Path(temporary)
            with SAM2Client(
                sys.executable,
                workspace,
                models_dir,
                checkpoint=checkpoint,
                config=config,
            ) as segmenter:
                tracker = cls(
                    segmenter,
                    loaded,
                    foundationpose_root=workspace / "third_party/FoundationPose",
                    debug_dir=debug_dir / "foundationpose",
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
        for tracker in self.trackers.values():
            tracker.estimator.pose_last = None

    def detect_masks(self, frame, prompts):
        candidates = self.segmenter.predict(frame.rgb, ". ".join(prompts.values()) + ".")
        masks = {}
        for name, prompt in prompts.items():
            matches = [item for item in candidates if item["label"].strip().lower().rstrip(".") == prompt.strip().lower().rstrip(".")]
            if len(matches) != 1:
                raise TrackingFailure(f"{name}: 首帧候选数 {len(matches)}，无法无歧义初始化")
            masks[name] = matches[0]["mask"]
        return masks

    def step(self, frame, *, initial_masks=None) -> dict[str, np.ndarray]:
        """返回完整物体位姿；预期失败抛 TrackingFailure，本段不得继续。"""
        maps = self.maps
        first = maps is None
        if first:
            if not isinstance(initial_masks, dict) or initial_masks.keys() != self.trackers.keys():
                raise ValueError("The first frame requires one initial mask per Mesh")
        elif initial_masks is not None:
            raise ValueError("Call reset() before initializing another scene")
        if maps is None:
            validate_intrinsics(frame.K)
            validate_distortion(frame.distortion_model, frame.distortion_coeffs)
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
            # 排除无明确归属的重叠像素，在 IPC 前识别场景初始化失败。
            unique = np.stack(list(masks.values())).sum(axis=0) == 1
            masks = {name: mask & unique for name, mask in masks.items()}
            for name, mask in masks.items():
                if not mask.any():
                    raise TrackingFailure(f"{name}: initial mask empty after rectification/overlap exclusion")
            masks = self.segmenter.initialize(rgb, masks)
            self.maps = maps
            self.directions = camera_rays(depth_m.shape, K, PIXEL_STRIDE)
        else:
            masks = self.segmenter.track(rgb)
        sampled_depth = depth_m[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        poses = {}
        prepared_frame = None
        for name, tracker in self.trackers.items():
            mask = masks[name]
            selected = mask[::PIXEL_STRIDE, ::PIXEL_STRIDE] & (sampled_depth > 0)
            observed_depth = sampled_depth[selected]
            count = len(observed_depth)
            fp_count = np.count_nonzero(mask & (depth_m >= 0.001))
            if count < MIN_MATCHED_PIXELS or fp_count < MIN_DEPTH_PIXELS:
                raise TrackingFailure(
                    f"{name}: insufficient segmented depth: {count} independent points, "
                    f"{fp_count} FoundationPose pixels; requires "
                    f"{MIN_MATCHED_PIXELS} and {MIN_DEPTH_PIXELS}"
                )
            if not first and prepared_frame is None:
                prepared_frame = prepare_tracking_frame(rgb, depth_m, K)
            try:
                pose = (
                    tracker.register(rgb, depth_m, K, mask)
                    if first
                    else tracker.track(K, mask, prepared_frame=prepared_frame)
                )
            except TrackingFailure as error:
                raise TrackingFailure(f"{name}: {error}") from error
            v, u = np.nonzero(selected)
            pixels = np.column_stack((v * PIXEL_STRIDE, u * PIXEL_STRIDE)).astype(np.int32)
            model_z = self.renderers[name].render(pixels, pose, directions=self.directions)
            supported = np.count_nonzero(
                (model_z > 0) & (np.abs(observed_depth - model_z) <= DEPTH_TOLERANCE_M)
            )
            if supported < MIN_MATCHED_PIXELS or supported / count < MIN_MATCHED_FRACTION:
                raise TrackingFailure(f"{name}: Semantic depth/model disagreement; pose lost")
            poses[name] = pose
        return poses

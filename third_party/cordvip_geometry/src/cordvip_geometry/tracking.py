"""语义分割与深度门控的多物体位姿跟踪。"""

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
DEPTH_TOLERANCE_M = 0.015
MIN_MATCHED_PIXELS = 30
MIN_MATCHED_FRACTION = 0.15


class SceneTracker:
    def __init__(self, mesh_paths, *, workspace):
        from .segmentation import SAM2Segmenter
        from .weights import DEFAULT_MODELS_DIR, resolve_sam_model

        workspace = Path(workspace)
        models_dir = workspace / DEFAULT_MODELS_DIR
        checkpoint, config = resolve_sam_model(models_dir)
        self.segmenter = SAM2Segmenter(
            workspace / "third_party/Grounded-SAM-2",
            checkpoint,
            config,
            models_dir / "grounding-dino-base",
        )
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

    def initialize(self, frame, prompts):
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
                    f"{name}: 首帧候选数 {len(matches)}，无法无歧义初始化"
                )
            boxes[name] = matches[0]["box_xyxy"]

        masks = self.segmenter.initialize(rgb, boxes)
        return self._estimate(rgb, depth_m, frame.K, masks, first=True)

    def track(self, frame):
        rgb, depth_m = self.rectifier.apply(frame.rgb, frame.depth_m)
        masks = self.segmenter.track(rgb)
        return self._estimate(rgb, depth_m, frame.K, masks, first=False)

    def _estimate(self, rgb, depth_m, K, masks, *, first) -> dict[str, np.ndarray]:
        """返回各物体局部坐标到相机坐标的位姿。"""
        sampled_depth = depth_m[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        depth_masks = {}
        for name in self.trackers:
            mask = masks[name]
            selected = mask[::PIXEL_STRIDE, ::PIXEL_STRIDE] & (sampled_depth > 0)
            sampled_count = np.count_nonzero(selected)
            depth_count = np.count_nonzero(mask & (depth_m >= 0.001))
            if sampled_count < MIN_MATCHED_PIXELS or depth_count < MIN_DEPTH_PIXELS:
                raise TrackingFailure(
                    f"{name}: insufficient segmented depth: {sampled_count} independent points, "
                    f"{depth_count} FoundationPose pixels; requires "
                    f"{MIN_MATCHED_PIXELS} and {MIN_DEPTH_PIXELS}"
                )
            depth_masks[name] = selected

        poses = {} if first else track_objects(self.trackers, rgb, depth_m, K, masks)
        for name, tracker in self.trackers.items():
            if first:
                try:
                    poses[name] = tracker.register(rgb, depth_m, K, masks[name])
                except TrackingFailure as error:
                    raise TrackingFailure(f"{name}: {error}") from error

            selected = depth_masks[name]
            observed_depth = sampled_depth[selected]
            model_z = self.renderers[name].render(
                self.directions[selected], poses[name]
            )
            supported = np.count_nonzero(
                (model_z > 0) & (np.abs(observed_depth - model_z) <= DEPTH_TOLERANCE_M)
            )
            if (
                supported < MIN_MATCHED_PIXELS
                or supported / len(observed_depth) < MIN_MATCHED_FRACTION
            ):
                raise TrackingFailure(
                    f"{name}: Semantic depth/model disagreement; pose lost"
                )
        return poses

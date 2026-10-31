"""Raw 同行分割、实测点云与可选位姿；逐行结果直接编码。"""

from pathlib import Path
import logging

import numpy as np

from .geometry import transform_points
from .foundationpose import prepare_tracking_frame
from .model_depth import ModelDepthRenderer, camera_rays
from .scene_geometry import extract_scene_clouds, add_model_depth_support

logger = logging.getLogger(__name__)


class SceneTracker:
    """共享模型权重；无资产时保留实测点，位姿丢失持续到新会话显式注册。"""

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
    ):
        from .foundationpose import FoundationPosePlusTracker

        self.segmenter = segmenter
        self.assets = assets
        self.pixel_stride = pixel_stride
        self.depth_tolerance_m = depth_tolerance_m
        self.min_matched_pixels = min_matched_pixels
        self.min_matched_fraction = min_matched_fraction
        self.trackers, self.renderers = {}, {}
        self.lost_objects = {}
        self.instance_ids = self.previous_frame_identity = self.image_shape = self.intrinsics = None
        self.directions = None
        shared = None
        for name, (mesh, points, normals) in self.assets.items():
            tracker = FoundationPosePlusTracker(
                foundationpose_root, mesh, Path(debug_dir) / name, shared_with=shared
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = ModelDepthRenderer(mesh, pixel_stride=pixel_stride)

    def step(self, frame, rgb, depth_m, K, *, initial_masks=None):
        identity = (frame.source_id, frame.row_index, frame.clock, frame.timestamp_ns)
        first = self.instance_ids is None
        if first != (initial_masks is not None):
            raise ValueError("Supply initial_masks exactly once, on the first scene frame")
        if not first:
            if (
                identity == self.previous_frame_identity
                or frame.clock != self.previous_frame_identity[2]
                or frame.timestamp_ns < self.previous_frame_identity[3]
            ):
                raise ValueError("Scene requires chronological, distinct frames on one clock")
            if (frame.row_index is None) != (self.previous_frame_identity[1] is None) or (
                frame.row_index is not None and frame.row_index <= self.previous_frame_identity[1]
            ):
                raise ValueError(
                    "Scene source row must advance; explicitly start a new session on replay reset"
                )
            if rgb.shape != self.image_shape or not np.array_equal(K, self.intrinsics):
                raise ValueError(
                    "Scene image shape/intrinsics changed; explicitly start a new session"
                )
        elif not set(self.assets).issubset(initial_masks):
            raise ValueError("Every pose asset requires a named initial instance mask")
        frame_key = dict(
            source_id=frame.source_id,
            row_index=frame.row_index,
            clock=frame.clock,
            timestamp_ns=frame.timestamp_ns,
        )
        masks = (
            self.segmenter.initialize(rgb, initial_masks, frame_key=frame_key)
            if first
            else self.segmenter.track(rgb, frame_key=frame_key)
        )
        instance_ids = tuple(initial_masks) if first else self.instance_ids
        if tuple(masks) != instance_ids:
            raise RuntimeError("Streaming segmentation changed instance identity/order")
        # Independent evidence survives pose failure or the absence of an asset.
        measured = extract_scene_clouds(rgb, depth_m, K, masks, pixel_stride=self.pixel_stride)
        if first:
            for name in self.assets:
                count = len(measured.objects[name].observed_points_camera)
                if count < self.min_matched_pixels:
                    raise ValueError(
                        f"{name}: pose initialization has {count} independent depth points "
                        f"after segmentation/overlap/stride; requires {self.min_matched_pixels}. "
                        "Select a supported instance and start a new session."
                    )
            self.instance_ids = instance_ids
            self.image_shape, self.intrinsics = rgb.shape, K.copy()
            self.directions = camera_rays(depth_m.shape, K, self.pixel_stride) if self.assets else None
        poses, depths = {}, {}
        prepared_frame = None
        for name, tracker in self.trackers.items():
            if name in self.lost_objects:
                continue
            cloud = measured.objects[name]
            if len(cloud.observed_points_camera) < self.min_matched_pixels:
                self.lost_objects[name] = (
                    "Insufficient independently segmented depth; pose lost"
                )
                continue
            try:
                mask = masks[name] & ~measured.overlap_mask
                if not first and prepared_frame is None:
                    prepared_frame = prepare_tracking_frame(rgb, depth_m, K)
                poses[name] = (
                    tracker.register(rgb, depth_m, K, mask)
                    if first
                    else tracker.track(rgb, depth_m, K, mask, prepared_frame=prepared_frame)
                )
                depths[name] = self.renderers[name].render(
                    depth_m.shape, K, poses[name], directions=self.directions,
                )
            except Exception as error:
                logger.exception("%s pose failed at Raw row %s (%s)", name, frame.row_index, frame.source_id)
                poses.pop(name, None)
                self.lost_objects[name] = f"{type(error).__name__}: {error}"
        add_model_depth_support(measured, depths, depth_tolerance_m=self.depth_tolerance_m)
        objects = {}
        for name in self.instance_ids:
            cloud = measured.objects[name]
            pose = poses.get(name)
            supported = cloud.diagnostics["model_supported_point_count"] or 0
            fraction = cloud.diagnostics["model_supported_fraction"] or 0.0
            if pose is not None and (
                supported < self.min_matched_pixels or fraction < self.min_matched_fraction
            ):
                self.lost_objects[name] = "Semantic depth/model disagreement; pose lost"
                pose = None
            pose_valid = pose is not None
            points = (
                transform_points(self.assets[name][1], pose)
                if pose_valid
                else np.empty((0, 3), np.float32)
            )
            normals = (
                self.assets[name][2] @ pose[:3, :3].T
                if pose_valid
                else np.empty((0, 3), np.float32)
            )
            objects[name] = dict(
                cloud=cloud,
                pose=pose,
                pose_valid=pose_valid,
                model_points_camera=points,
                model_normals_camera=normals,
                state=(
                    "TRACKING"
                    if pose_valid
                    else "POSE_LOST" if name in self.assets else "OBSERVED_ONLY"
                ),
                error=self.lost_objects.get(name),
                diagnostic_pose=poses.get(name),
                observed_valid=bool(len(cloud.observed_points_camera)),
            )
        self.previous_frame_identity = identity
        return dict(
            frame=frame,
            rgb=rgb,
            K=K,
            objects=objects,
            operation="initialize" if first else "track",
        )

    def close(self):
        self.segmenter.close()
        self.trackers.clear()
        self.renderers.clear()


def scene_arrays(result, hand=None):
    """仅编码当前已处理 Raw 行；对象角色保存在一次性的 scene.json。"""
    frame = result["frame"]
    payload = dict(
        source_id=frame.source_id,
        row_index=frame.row_index,
        timestamp_ns=frame.timestamp_ns,
        clock=frame.clock,
        rgb=result["rgb"],
        K=result["K"],
    )
    if hand is not None:
        T_base_camera = hand["T_base_camera"]
        hand_points, hand_normals = hand["points_camera"], hand["normals_camera"]
        payload.update(
            hand_points_camera=hand_points,
            hand_normals_camera=hand_normals,
            hand_points_base=transform_points(hand_points, T_base_camera),
            hand_normals_base=(hand_normals @ T_base_camera[:3, :3].T).astype(np.float32),
            hand_link_ids=hand["link_ids"],
            joint_state=hand["joint_state"],
            T_base_camera=T_base_camera,
            hand_timing_available=hand["timing_available"],
            hand_camera_skew_ms=hand["skew_ms"] if hand["timing_available"] else np.nan,
            arm_timestamp_ns=hand["arm_timestamp_ns"],
            hand_timestamp_ns=hand["hand_timestamp_ns"],
            camera_timestamp_ns=hand["camera_timestamp_ns"],
        )
    for name, item in result["objects"].items():
        cloud, pose_ok = item["cloud"], item["pose_valid"]
        fields = dict(
            state=item["state"],
            pose_valid=pose_ok,
            error=item["error"] or "",
            observed_valid=item["observed_valid"],
            T_camera_object=item["pose"].copy() if pose_ok else np.full((4, 4), np.nan),
            model_points_camera=item["model_points_camera"],
            model_normals_camera=item["model_normals_camera"],
            observed_points_camera=cloud.observed_points_camera,
            observed_colors=cloud.observed_colors,
            semantic_mask=cloud.semantic_mask,
            observed_mask=cloud.observed_mask,
        )
        if hand is not None:
            fields["observed_points_base"] = transform_points(
                fields["observed_points_camera"], T_base_camera
            )
            fields["model_points_base"] = transform_points(
                fields["model_points_camera"], T_base_camera
            )
        if pose_ok and cloud.model_supported_mask is not None:
            fields["model_supported_mask"] = cloud.model_supported_mask
            fields["observed_model_depth_residual_m"] = cloud.observed_model_depth_residual_m
        payload.update({f"{name}__{key}": value for key, value in fields.items()})
    return payload

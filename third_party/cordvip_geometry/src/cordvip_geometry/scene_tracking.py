"""Raw 同行分割、实测点云与可选位姿；逐行结果直接编码。"""

from pathlib import Path

import numpy as np

from .foundationpose import MIN_DEPTH_PIXELS, prepare_tracking_frame
from .model_depth import ModelDepthRenderer, camera_rays
from .scene_geometry import add_model_depth_support, extract_scene_clouds


class SceneTracker:
    """一个已校验 Raw 流的内部 tracker；共享权重，位姿丢失后仍保留实测点。"""

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
        self.pixel_stride = pixel_stride
        self.depth_tolerance_m = depth_tolerance_m
        self.min_matched_pixels = min_matched_pixels
        self.min_matched_fraction = min_matched_fraction
        self.trackers, self.renderers = {}, {}
        self.lost_objects = {}
        self.directions = None
        shared = None
        for name, (mesh, _, _) in assets.items():
            tracker = FoundationPosePlusTracker(
                foundationpose_root, mesh, Path(debug_dir) / name, shared_with=shared
            )
            shared = shared or tracker
            self.trackers[name] = tracker
            self.renderers[name] = ModelDepthRenderer(mesh, pixel_stride=pixel_stride)

    def step(self, frame, rgb, depth_m, *, initial_masks=None):
        K = frame.K
        first = initial_masks is not None
        if first and not set(self.trackers).issubset(initial_masks):
            raise ValueError("Every pose asset requires a named initial instance mask")
        masks = (
            self.segmenter.initialize(rgb, initial_masks)
            if first
            else self.segmenter.track(rgb)
        )
        if first:
            self.directions = camera_rays(depth_m.shape, K, self.pixel_stride)
        # Independent evidence survives pose failure or the absence of an asset.
        measured = extract_scene_clouds(rgb, depth_m, self.directions, masks, pixel_stride=self.pixel_stride)
        poses, model_z_by_object = {}, {}
        prepared_frame = None
        for name, tracker in self.trackers.items():
            if name in self.lost_objects:
                continue
            cloud = measured[name]
            mask = masks[name]
            count = len(cloud.observed_points_camera)
            fp_count = np.count_nonzero(mask & (depth_m >= 0.001))
            if count < self.min_matched_pixels or fp_count < MIN_DEPTH_PIXELS:
                if first:
                    raise ValueError(
                        f"{name}: pose initialization has {count} independent depth points "
                        f"after segmentation/stride and {fp_count} FoundationPose depth pixels; "
                        f"requires {self.min_matched_pixels} and {MIN_DEPTH_PIXELS}, respectively. "
                        "Select a supported instance and start a new session."
                    )
                self.lost_objects[name] = (
                    "Insufficient independently segmented depth; pose lost"
                )
                continue
            if not first and prepared_frame is None:
                prepared_frame = prepare_tracking_frame(rgb, depth_m, K)
            poses[name] = (
                tracker.register(rgb, depth_m, K, mask)
                if first
                else tracker.track(rgb, depth_m, K, mask, prepared_frame=prepared_frame)
            )
            model_z_by_object[name] = self.renderers[name].render(
                cloud.observed_pixel_indices, K, poses[name], directions=self.directions,
            )
        add_model_depth_support(measured, model_z_by_object, depth_tolerance_m=self.depth_tolerance_m)
        objects = {}
        for name, cloud in measured.items():
            pose = poses.get(name)
            supported = cloud.model_supported_point_count
            fraction = cloud.model_supported_fraction
            if pose is not None and (
                supported < self.min_matched_pixels or fraction < self.min_matched_fraction
            ):
                self.lost_objects[name] = "Semantic depth/model disagreement; pose lost"
                pose = None
            pose_valid = pose is not None
            objects[name] = dict(
                cloud=cloud,
                pose=pose,
                pose_valid=pose_valid,
                state=(
                    "TRACKING"
                    if pose_valid
                    else "POSE_LOST" if name in self.trackers else "OBSERVED_ONLY"
                ),
                error=self.lost_objects.get(name),
                candidate_pose=poses.get(name),
                observed_valid=bool(len(cloud.observed_points_camera)),
            )
        return dict(
            frame=frame,
            rgb=rgb,
            objects=objects,
            operation="initialize" if first else "track",
        )

    def close(self):
        self.trackers.clear()
        self.renderers.clear()


def scene_arrays(result, hand=None):
    """编码当前 Raw 行；固定模型保存在 model_templates，模型逐行仅保存位姿。"""
    frame = result["frame"]
    payload = dict(
        source_id=frame.source_id,
        row_index=frame.row_index,
        timestamp_ns=frame.timestamp_ns,
        clock=frame.clock,
        rgb=result["rgb"],
        K=frame.K,
        operation=result["operation"],
    )
    if hand is not None:
        T_base_camera = hand["T_base_camera"]
        hand_points, hand_normals = hand["points_camera"], hand["normals_camera"]
        payload.update(
            hand_points_camera=hand_points,
            hand_normals_camera=hand_normals,
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
            T_camera_object=item["pose"] if pose_ok else np.full((4, 4), np.nan),
            candidate_T_camera_object=(
                item["candidate_pose"] if item["candidate_pose"] is not None else np.full((4, 4), np.nan)
            ),
            observed_points_camera=cloud.observed_points_camera,
            observed_colors=cloud.observed_colors,
            semantic_mask=cloud.semantic_mask,
            observed_mask=cloud.observed_mask,
        )
        if cloud.model_supported_mask is not None:
            fields["model_supported_mask"] = cloud.model_supported_mask
            fields["observed_model_depth_residual_m"] = cloud.observed_model_depth_residual_m
            fields["model_supported_point_count"] = cloud.model_supported_point_count
            fields["model_supported_fraction"] = cloud.model_supported_fraction
        payload.update({f"{name}__{key}": value for key, value in fields.items()})
    return payload

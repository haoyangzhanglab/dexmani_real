"""Same-frame instance masks, measured surfaces and optional independent poses.

The host owns acquisition and timestamps. This module opens no hardware, performs
no temporal interpolation, and never promotes generated geometry to observation.
"""

from pathlib import Path
import time

import numpy as np

from .geometry import _float32_points, transform_points, validate_pose, validate_rgbd
from .object_cloud import ObjectCloudProjector
from .observation import _normals
from .scene_geometry import extract_scene_clouds


class SceneTracker:
    """One sequential GPU owner; shared segmentation and pose network weights.

    ``assets`` maps fixed instance IDs to (mesh, points, normals). IDs without an
    asset still produce semantic depth clouds. Pose loss is latched per object;
    constructing a new session explicitly registers again, with new first masks.
    """

    def __init__(self, segmenter, assets, *, foundationpose_root, debug_dir,
                 pixel_stride=2, depth_tolerance_m=.015, min_matched_pixels=30,
                 min_matched_fraction=.15):
        from .foundationpose import FoundationPoseTracker

        self.segmenter = segmenter
        self.assets = assets
        self.pixel_stride = pixel_stride
        self.depth_tolerance_m = depth_tolerance_m
        self.min_matched_pixels = min_matched_pixels
        self.min_matched_fraction = min_matched_fraction
        self.trackers, self.projectors = {}, {}
        self._lost = {}
        self._ids = self._previous = self._shape = self._K = None
        shared = None
        for name, (mesh, points, normals) in assets.items():
            tracker = FoundationPoseTracker(
                foundationpose_root, mesh, Path(debug_dir) / name, shared_with=shared)
            shared = shared or tracker
            self.trackers[name] = tracker
            self.projectors[name] = ObjectCloudProjector(
                mesh, pixel_stride=pixel_stride, depth_tolerance_m=depth_tolerance_m)

    @staticmethod
    def _sync():
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def step(self, frame, rgb, depth_m, K, *, initial_masks=None):
        rgb, depth_m, K, _ = validate_rgbd(rgb, depth_m, K)
        identity = (frame.source_id, frame.row_index, frame.clock, frame.timestamp_ns)
        first = self._ids is None
        if first != (initial_masks is not None):
            raise ValueError('Supply initial_masks exactly once, on the first scene frame')
        if not first:
            if (identity == self._previous or frame.clock != self._previous[2]
                    or frame.timestamp_ns < self._previous[3]):
                raise ValueError('Scene requires chronological, distinct frames on one clock')
            if ((frame.row_index is None) != (self._previous[1] is None)
                    or (frame.row_index is not None and frame.row_index <= self._previous[1])):
                raise ValueError('Scene source row must advance; explicitly start a new session on replay reset')
            if rgb.shape != self._shape or not np.array_equal(K, self._K):
                raise ValueError('Scene image shape/intrinsics changed; explicitly start a new session')
        elif not set(self.assets).issubset(initial_masks):
            raise ValueError('Every pose asset requires a named initial instance mask')
        started_ns = time.monotonic_ns()
        masks = (self.segmenter.initialize(rgb, initial_masks) if first
                 else self.segmenter.track(rgb))
        self._sync()
        segmentation_ns = time.monotonic_ns()
        if first:
            self._ids = tuple(initial_masks)
            self._shape, self._K = rgb.shape, K.copy()
        if tuple(masks) != self._ids:
            raise RuntimeError('Streaming segmentation changed instance identity/order')
        # Independent evidence survives pose failure or the absence of an asset.
        measured = extract_scene_clouds(rgb, depth_m, K, masks, pixel_stride=self.pixel_stride)
        poses, depths, pose_ms, reasons = {}, {}, {}, {}
        for name, tracker in self.trackers.items():
            start = time.monotonic_ns()
            if name in self._lost:
                reasons[name] = self._lost[name]
                continue
            cloud = measured.objects[name]
            if len(cloud.observed_points_camera) < self.min_matched_pixels:
                self._lost[name] = reasons[name] = 'Insufficient independently segmented depth; pose lost'
                continue
            try:
                mask = masks[name] & ~measured.overlap_mask
                poses[name] = (tracker.register(rgb, depth_m, K, mask) if first
                               else tracker.track(rgb, depth_m, K))
                self._sync()
                pose_ms[name] = (time.monotonic_ns() - start) / 1e6
                depths[name] = self.projectors[name].project(rgb, depth_m, K, poses[name]).model_depth_m
            except Exception as error:
                poses.pop(name, None)
                self._lost[name] = reasons[name] = f'{type(error).__name__}: {error}'
        clouds = extract_scene_clouds(
            rgb, depth_m, K, masks, pixel_stride=self.pixel_stride,
            model_depths=depths, depth_tolerance_m=self.depth_tolerance_m)
        objects = {}
        for name in self._ids:
            cloud = clouds.objects[name]
            pose = poses.get(name)
            supported = (int(np.count_nonzero(cloud.model_supported_mask))
                         if cloud.model_supported_mask is not None else 0)
            fraction = supported / max(1, len(cloud.observed_points_camera))
            if pose is not None and (supported < self.min_matched_pixels
                                     or fraction < self.min_matched_fraction):
                self._lost[name] = reasons[name] = 'Semantic depth/model disagreement; pose lost'
                pose = None
            pose_valid = pose is not None
            points = (transform_points(self.assets[name][1], pose) if pose_valid
                      else np.empty((0, 3), np.float32))
            normals = (self.assets[name][2] @ pose[:3, :3].T if pose_valid
                       else np.empty((0, 3), np.float32))
            objects[name] = dict(
                cloud=cloud, pose=pose, pose_valid=pose_valid,
                model_points_camera=points, model_normals_camera=normals,
                state=('TRACKING' if pose_valid else 'POSE_LOST' if name in self.assets
                       else 'OBSERVED_ONLY'), error=reasons.get(name),
                diagnostic_pose=poses.get(name),
                observed_valid=bool(len(cloud.observed_points_camera)),
                semantic_model_support_fraction=fraction if name in depths else None,
            )
        self._previous = identity
        return dict(frame=frame, rgb=rgb, depth_m=depth_m, K=K, objects=objects,
                    overlap_mask=clouds.overlap_mask, started_ns=started_ns,
                    completed_ns=time.monotonic_ns(), operation='initialize' if first else 'track',
                    segmentation_ms=(segmentation_ns - started_ns) / 1e6, pose_ms=pose_ms)

    def close(self):
        self.segmenter.close()
        self.trackers.clear()
        self.projectors.clear()


def scene_payload(result, roles, *, valid, state, max_age_ms, hand=None):
    """Plain array snapshot; each object independently declares pose/observation validity.

    Instance names prefix arrays (``toy__...``); no pickle or object arrays. An
    invalid session snapshot contains no usable geometry. A lost pose does not
    suppress independent same-frame semantic depth points.
    """
    if (isinstance(max_age_ms, (bool, np.bool_))
            or not isinstance(max_age_ms, (int, float, np.integer, np.floating))
            or not np.isfinite(max_age_ms) or max_age_ms <= 0):
        raise ValueError('max_age_ms must be finite and positive')
    if valid and (result is None or result.get('frame') is None):
        raise ValueError('A valid scene observation requires a result and source frame')
    frame = result['frame'] if result else None
    payload = dict(
        valid=bool(valid), state=state, units='m', coordinate_frame='color_camera',
        object_ids=np.asarray(list(roles), dtype=str), object_roles=np.asarray(list(roles.values()), dtype=str),
        source_id=frame.source_id if frame else '', row_index=frame.row_index if frame and frame.row_index is not None else -1,
        timestamp_ns=frame.timestamp_ns if frame else -1, clock=frame.clock if frame else '',
        completed_ns=result['completed_ns'] if result else -1,
        published_ns=time.monotonic_ns(), max_age_ms=max_age_ms,
        K=result['K'] if result else np.full((3, 3), np.nan),
    )
    empty = np.empty((0, 3), np.float32)
    if hand is not None and valid:
        if (hand['source_id'] != frame.source_id or hand['row_index'] != frame.row_index
                or hand['clock'] != frame.clock):
            raise ValueError('Hand and scene must refer to exactly the same source row and clock')
        T_base_camera = validate_pose(hand['T_base_camera'])
        hand_points = _float32_points(hand['points_camera'])
        hand_normals = _normals(hand['normals_camera'], hand_points)
        link_ids = np.asarray(hand['link_ids'])
        joints = np.asarray(hand['joint_state'])
        if (not len(hand_points) or link_ids.shape != (len(hand_points),)
                or not np.issubdtype(link_ids.dtype, np.integer) or np.any(link_ids < 0)):
            raise ValueError('Hand link IDs must correspond to nonempty sampled points')
        if (joints.ndim != 1 or not len(joints) or joints.dtype.kind not in 'fiu'
                or not np.isfinite(joints).all()):
            raise ValueError('Joint state must contain finite measured joint positions')
        stamps = [hand[key] for key in ('arm_timestamp_ns', 'hand_timestamp_ns', 'camera_timestamp_ns')]
        if any(isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, (int, np.integer)) for stamp in stamps):
            raise ValueError('Source timestamps must be integer nanoseconds')
        if not isinstance(hand['timing_available'], (bool, np.bool_)):
            raise ValueError('timing_available must be a boolean source-evidence flag')
        timing = bool(hand['timing_available'])
        skew_ms = np.nan
        if timing:
            if min(stamps) <= 0 or stamps[2] != frame.timestamp_ns:
                raise ValueError('Available timing requires positive source stamps and matching camera time')
            skew_ms = (max(stamps) - min(stamps)) / 1e6
            supplied_skew = hand['skew_ms']
            if (isinstance(supplied_skew, (bool, np.bool_))
                    or not isinstance(supplied_skew, (int, float, np.integer, np.floating))
                    or not np.isfinite(supplied_skew)
                    or not np.isclose(supplied_skew, skew_ms, rtol=0, atol=1e-9)):
                raise ValueError('Hand/camera skew must be calculated from original source times')
        payload.update(hand_points_camera=hand_points, hand_normals_camera=hand_normals,
                       hand_points_base=transform_points(hand_points, T_base_camera),
                       hand_normals_base=(hand_normals @ T_base_camera[:3, :3].T).astype(np.float32),
                       hand_link_ids=link_ids, joint_state=joints,
                       T_base_camera=T_base_camera, hand_timing_available=timing,
                       hand_camera_skew_ms=skew_ms,
                       arm_timestamp_ns=stamps[0], hand_timestamp_ns=stamps[1], camera_timestamp_ns=stamps[2])
    for name in roles:
        item = result['objects'][name] if result and valid else None
        cloud = item['cloud'] if item else None
        pose_ok = bool(item and item['pose_valid'])
        fields = dict(
            state=item['state'] if item else state, pose_valid=pose_ok,
            error=str(item['error']) if item and item.get('error') is not None else '',
            observed_valid=bool(item and item['observed_valid']),
            T_camera_object=validate_pose(item['pose']) if pose_ok else np.full((4, 4), np.nan),
            model_points_camera=item['model_points_camera'] if pose_ok else empty,
            model_normals_camera=item['model_normals_camera'] if pose_ok else empty,
            observed_points_camera=cloud.observed_points_camera if cloud else empty,
            observed_colors=cloud.observed_colors if cloud else np.empty((0, 3), np.uint8),
            semantic_mask=cloud.semantic_mask if cloud else np.empty((0, 0), bool),
            observed_mask=cloud.observed_mask if cloud else np.empty((0, 0), bool),
        )
        if hand is not None and valid:
            fields['observed_points_base'] = transform_points(fields['observed_points_camera'], T_base_camera)
            fields['model_points_base'] = transform_points(fields['model_points_camera'], T_base_camera)
        if pose_ok and cloud is not None and cloud.model_supported_mask is not None:
            fields['model_supported_mask'] = cloud.model_supported_mask
            fields['observed_model_depth_residual_m'] = cloud.observed_model_depth_residual_m
        payload.update({f'{name}__{key}': value for key, value in fields.items()})
    return payload

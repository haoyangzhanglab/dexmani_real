"""Build in-memory hand/object observations; disk publication is optional.

Geometry validity is separate from timing evidence. The caller controls source
age admission; an offline replay packet must never authorize robot motion.
"""

import json
from pathlib import Path
import time

import numpy as np

from .geometry import _float32_points, transform_points, validate_pose


def _normals(value, points):
    normals = _float32_points(value)
    if normals.shape != points.shape or not np.allclose(
        np.linalg.norm(normals, axis=1), 1., rtol=0, atol=1e-4
    ):
        raise ValueError('Normals must be unit vectors corresponding to every model point')
    return normals


def build_observation(*, state, valid, result, cloud, model_points, max_age_ms,
                      hand=None, model_normals=None):
    """Return a plain NumPy payload without importing devices, models or a viewer.

    Model points/normals and optional hand geometry arrive in the color-camera
    frame (metres). Base-frame outputs use the supplied T_base_camera. Hand
    joints must be measured, associated with the same source row, and ordered as
    declared in the provider's run metadata; no interpolation is performed.

    Arrays are borrowed for immediate consumption. A retaining/async consumer
    should copy its inputs and enforce its own freshness/skew requirements.
    ``interaction_valid`` means geometry is available, including offline rows
    without timing evidence; it is never a policy synchronization/admission flag.
    """
    if not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError('max_age_ms must be finite and positive')
    if valid and (result is None or result.pose is None or cloud is None or model_points is None):
        raise ValueError('Valid observation requires pose, source frame and object clouds')
    frame = result.frame if result is not None else None
    empty = np.empty((0, 3), np.float32)
    missing_pose = np.full((4, 4), np.nan)
    payload = dict(
        valid=bool(valid), state=state, units='m', coordinate_frame='color_camera',
        source_id=frame.source_id if frame else '',
        row_index=frame.row_index if frame and frame.row_index is not None else -1,
        timestamp_ns=frame.timestamp_ns if frame else -1, clock=frame.clock if frame else '',
        completed_ns=result.completed_ns if result else -1,
        published_ns=time.monotonic_ns(), max_age_ms=max_age_ms,
        T_camera_object=validate_pose(result.pose) if valid else missing_pose,
        model_points_camera=_float32_points(model_points) if valid else empty,
        model_normals_camera=empty,
        observed_points_camera=_float32_points(cloud.observed_points_camera) if valid else empty,
        observed_colors=cloud.observed_colors if valid else np.empty((0, 3), np.uint8),
        K=result.K if result else np.full((3, 3), np.nan),
        diagnostics_json=json.dumps(cloud.diagnostics if cloud else {}, allow_nan=False),
        interaction_valid=False, base_coordinate_frame='',
        T_base_camera=missing_pose, T_base_object=missing_pose,
        object_points_base=empty, object_normals_base=empty,
        hand_points_camera=empty, hand_normals_camera=empty,
        hand_points_base=empty, hand_normals_base=empty,
        hand_link_ids=np.empty(0, np.int32), joint_state=np.empty(0, np.float64),
        hand_timing_available=False, hand_camera_skew_ms=np.nan,
        arm_timestamp_ns=-1, hand_timestamp_ns=-1, camera_timestamp_ns=-1,
    )
    if not valid:
        return payload
    if model_normals is not None:
        payload['model_normals_camera'] = _normals(model_normals, payload['model_points_camera'])
    if hand is None:
        return payload
    if (hand['source_id'] != frame.source_id or hand['clock'] != frame.clock
            or hand['row_index'] != frame.row_index):
        raise ValueError('Hand and object geometry must refer to the same source row and clock')
    T_base_camera = validate_pose(hand['T_base_camera'])
    hand_points = _float32_points(hand['points_camera'])
    hand_normals = _normals(hand['normals_camera'], hand_points)
    link_ids = np.asarray(hand['link_ids'])
    joints = np.asarray(hand['joint_state'])
    if (not len(hand_points) or link_ids.shape != (len(hand_points),)
            or not np.issubdtype(link_ids.dtype, np.integer) or np.any(link_ids < 0)):
        raise ValueError('Hand link IDs must correspond to nonempty sampled points')
    if joints.ndim != 1 or not len(joints) or np.iscomplexobj(joints) or not np.isfinite(joints).all():
        raise ValueError('Joint state must contain finite measured joint positions')
    stamps = [hand[key] for key in ('arm_timestamp_ns', 'hand_timestamp_ns', 'camera_timestamp_ns')]
    timing = bool(hand['timing_available'])
    if timing:
        if min(stamps) <= 0 or stamps[2] != frame.timestamp_ns:
            raise ValueError('Available timing requires positive source stamps and matching camera time')
        skew_ms = (max(stamps) - min(stamps)) / 1e6
        if hand['skew_ms'] is None or not np.isclose(hand['skew_ms'], skew_ms, rtol=0, atol=1e-9):
            raise ValueError('Hand/camera skew must be calculated from original source times')
    else:
        skew_ms = np.nan
    rotation = T_base_camera[:3, :3]
    payload.update(
        interaction_valid=True, base_coordinate_frame='robot_base',
        T_base_camera=T_base_camera, T_base_object=T_base_camera @ payload['T_camera_object'],
        object_points_base=transform_points(payload['model_points_camera'], T_base_camera),
        object_normals_base=(payload['model_normals_camera'] @ rotation.T).astype(np.float32),
        hand_points_camera=hand_points, hand_normals_camera=hand_normals,
        hand_points_base=transform_points(hand_points, T_base_camera),
        hand_normals_base=(hand_normals @ rotation.T).astype(np.float32),
        hand_link_ids=link_ids, joint_state=joints,
        hand_timing_available=timing, hand_camera_skew_ms=skew_ms,
        arm_timestamp_ns=stamps[0], hand_timestamp_ns=stamps[1], camera_timestamp_ns=stamps[2],
    )
    return payload


def write_observation(path, payload):
    """Atomically publish a snapshot; consumers must recheck source freshness."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **payload)
    temporary.replace(path)

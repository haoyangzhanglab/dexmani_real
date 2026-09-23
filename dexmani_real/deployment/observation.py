"""Project real control-row history into the public PolicySpec observation mapping."""

from typing import Any

import numpy as np

from dexmani_real.deployment.config import FingertipAssemblerConfig
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base, make_arm_fk
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics


def _requested_observation_fields(policy_spec: Any) -> set[str]:
    """Return source names directly from the validated ordered Policy fields."""
    return {field.name for field in policy_spec.observation_fields}


def build_fingertip_runtime(
    policy_spec: Any,
    fingertip_config: FingertipAssemblerConfig | None,
) -> tuple[object, HandKinematics | None, FingertipAssemblerConfig | None] | None:
    """Construct the local FK resources only when the observation requests them.

    ``eef_pose`` needs just the cached canonical ``ArmFK``; the hand FK and
    mount config are built only for ``fingertip_points``.  Both derive from
    the aligned arm qpos history, never from a second realtime EEF source.
    """
    requested = _requested_observation_fields(policy_spec)
    needs_hand_fk = "fingertip_points" in requested
    if not needs_hand_fk and "eef_pose" not in requested:
        return None
    if not needs_hand_fk:
        return make_arm_fk(), None, None
    if not isinstance(fingertip_config, FingertipAssemblerConfig):
        raise TypeError("fingertip_points requires FingertipAssemblerConfig")
    hand_fk = HandKinematics(
        fingertip_config.hand_urdf_path,
        list(fingertip_config.fingertip_link_names),
    )
    if not hand_fk.is_ready():
        raise RuntimeError("fingertip FK startup failed")
    return make_arm_fk(), hand_fk, fingertip_config


def build_policy_observation(rows, policy_spec, *, fingertip_runtime=None):
    if not rows:
        return None
    requested = _requested_observation_fields(policy_spec)
    if any(row.hand is None for row in rows):
        return None
    for name, field in (
        ("contact_force", "tactile_aggregate_valid"),
        ("tactile_force", "tactile_dense_valid"),
    ):
        if name in requested and not all(bool(row.hand[field][0]) for row in rows):
            return None
    joint = np.stack([np.concatenate((r.arm["qpos"][0], r.hand["qpos"][0])) for r in rows]).astype(
        np.float32
    )
    arrays = {"joint_state": joint}
    for name, field in (("contact_force", "tactile_aggregate"), ("tactile_force", "tactile_dense")):
        if name in requested:
            arrays[name] = np.stack([r.hand[field][0] for r in rows]).astype(np.float32)
    if "point_cloud" in requested:
        if any(r.point_cloud is None for r in rows):
            return None
        arrays["point_cloud"] = np.stack([r.point_cloud for r in rows])
    if "rgb" in requested:
        images = [r.camera["rgb"] for r in rows]
        if any(
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
            for image in images
        ):
            raise ValueError("rgb must be raw uint8 HWC with three channels")
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("rgb history must have stackable raw image shapes")
        arrays["rgb"] = np.stack(images)
    if requested & {"eef_pose", "fingertip_points"}:
        arm_fk, hand_fk, cfg = fingertip_runtime
        poses = compute_eef_pose_history_xarm_base(joint[:, :7], arm_fk=arm_fk)
        arrays["eef_pose"] = poses.astype(np.float32)
        if "fingertip_points" in requested:
            arrays["fingertip_points"] = compute_fingertip_history_xarm_base(
                joint[:, :7],
                joint[:, 7:],
                hand_fk=hand_fk,
                handbase_position_eef_m=np.asarray(cfg.handbase_position_eef_m),
                handbase_quat_eef_wxyz=np.asarray(cfg.handbase_quat_eef_wxyz),
                arm_fk=arm_fk,
                eef_pose_history=poses,
            ).astype(np.float32)
    result = {}
    for field in policy_spec.observation_fields:
        values = np.ascontiguousarray(arrays[field.name])
        if (
            values.dtype != np.dtype(field.dtype)
            or (field.name != "rgb" and values.shape != (len(rows), *field.shape))
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"invalid policy observation {field.name}")
        result[field.name] = values
    return result

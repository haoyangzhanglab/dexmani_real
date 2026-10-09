"""Project real control-row history into the requested Policy observation mapping."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.planning.kinematics.arm_fk import (
    ArmFK,
    compute_eef_pose_history_xarm_base,
    make_arm_fk,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


@dataclass(frozen=True)
class ObservationKinematics:
    arm_fk: ArmFK
    hand_fk: HandKinematics | None = None
    mount: HandParams | None = None


def build_observation_kinematics(policy_info: Any, runtime: Any):
    """Construct the local FK resources only when the observation requests them.

    ``eef_pose`` needs just the cached canonical ``ArmFK``; the hand FK and
    mount config are built only for ``fingertip_points``.  Both derive from
    the aligned arm qpos history, never from a second realtime EEF source.
    """
    requested = set(policy_info.observation_fields)
    needs_hand_fk = "fingertip_points" in requested
    if not needs_hand_fk and "eef_pose" not in requested:
        return None
    if not needs_hand_fk:
        return ObservationKinematics(make_arm_fk())
    hand_fk = HandKinematics(
        str(XHAND_RIGHT_URDF_PATH),
        list(policy_info.fingertip_link_names),
    )
    return ObservationKinematics(make_arm_fk(), hand_fk, runtime.hand)


def policy_observation_issue(row, fields):
    """Check each real row, including bootstrap and slots using an existing plan."""
    if row is None or row.hand is None:
        return "required_observation_unavailable"
    if not np.isfinite(row.arm["qpos"]).all() or not np.isfinite(row.hand["qpos"]).all():
        return "control_feedback_nonfinite"
    for name in fields:
        if name in {"contact_force", "tactile_force"}:
            channel = "aggregate" if name == "contact_force" else "dense"
            if not bool(row.hand[f"tactile_{channel}_valid"][0]):
                return f"required_{name}_unavailable"
            if not np.isfinite(row.hand[f"tactile_{channel}"]).all():
                return f"required_{name}_nonfinite"
        elif name == "point_cloud":
            if row.point_cloud is None or not np.isfinite(row.point_cloud).all():
                return "required_point_cloud_unavailable_or_nonfinite"
        elif name == "rgb":
            if row.camera is None:
                return "required_rgb_unavailable"
            image = row.camera["rgb"]
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                return "required_rgb_invalid"
    return None


def build_policy_observation(rows, policy_info, *, kinematics=None):
    requested = set(policy_info.observation_fields)
    if not rows or any(row.hand is None for row in rows):
        return None
    for name, field in (
        ("contact_force", "tactile_aggregate_valid"),
        ("tactile_force", "tactile_dense_valid"),
    ):
        if name in requested and not all(bool(row.hand[field][0]) for row in rows):
            return None
    arrays = {}
    if requested & {"joint_state", "eef_pose", "fingertip_points"}:
        joint = np.stack(
            [np.concatenate((r.arm["qpos"][0], r.hand["qpos"][0])) for r in rows]
        ).astype(np.float32)
        arrays["joint_state"] = joint
    for name, field in (("contact_force", "tactile_aggregate"), ("tactile_force", "tactile_dense")):
        if name in requested:
            arrays[name] = np.stack([r.hand[field][0] for r in rows]).astype(np.float32)
    if "point_cloud" in requested:
        if any(r.point_cloud is None for r in rows):
            return None
        arrays["point_cloud"] = np.stack([r.point_cloud for r in rows])
    if "rgb" in requested:
        if any(r.camera is None for r in rows):
            return None
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
        arm_fk, hand_fk, cfg = kinematics.arm_fk, kinematics.hand_fk, kinematics.mount
        poses = compute_eef_pose_history_xarm_base(joint[:, :7], arm_fk=arm_fk)
        arrays["eef_pose"] = poses.astype(np.float32)
        if "fingertip_points" in requested:
            arrays["fingertip_points"] = compute_fingertip_history_xarm_base(
                joint[:, :7],
                joint[:, 7:],
                hand_fk=hand_fk,
                handbase_position_eef_m=np.asarray(cfg.T_eef_handbase_pos_xyz),
                handbase_quat_eef_wxyz=np.asarray(cfg.T_eef_handbase_quat_wxyz),
                arm_fk=arm_fk,
                eef_pose_history=poses,
            ).astype(np.float32)
    result = {}
    for name in policy_info.observation_fields:
        values = np.ascontiguousarray(arrays[name])
        if values.dtype != np.uint8 and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite policy observation {name}")
        result[name] = values
    return result


def policy_sources(row, fields):
    """Host source times for each requested field in the latest control row."""
    arm = int(row.arm["timestamp_ns"][0])
    hand = int(row.hand["timestamp_ns"][0])
    sources = {}
    for name in fields:
        if name in {"joint_state", "fingertip_points"}:
            times = (arm, hand)
        elif name == "eef_pose":
            times = (arm,)
        elif name in {"contact_force", "tactile_force"}:
            times = (hand,)
        elif name == "point_cloud":
            times = (row.pointcloud_timestamp_ns,)
        elif name == "rgb":
            times = (int(row.camera["timestamp_ns"]),)
        else:
            raise ValueError(f"No source clock for {name}")
        sources[name] = times
    return sources


def decision_is_fresh(sources, now_ns, max_age_s):
    return bool(sources) and all(
        0 < timestamp <= now_ns and now_ns - timestamp <= int(max_age_s * 1e9)
        for times in sources.values()
        for timestamp in times
    )

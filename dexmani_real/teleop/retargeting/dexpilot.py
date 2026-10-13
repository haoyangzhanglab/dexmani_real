"""DexPilot propagates technical solver failures instead of the upstream previous-target fallback."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import nlopt
import numpy as np
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.yourdfpy import DUMMY_JOINT_NAMES

from dexmani_real import ASSET_DIR
from dexmani_real.robot.model import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.retargeter import (
    _OPERATOR2MANO_RIGHT,
    _estimate_palm_frame,
    adaptive_retargeting_xhand,
    validate_landmarks,
)


logger = logging.getLogger(__name__)


class StrictDexPilotOptimizer(DexPilotOptimizer):
    """Propagate solver failures rather than substituting a previous target."""

    def retarget(
        self,
        ref_value: np.ndarray,
        fixed_qpos: np.ndarray,
        last_qpos: np.ndarray,
    ) -> np.ndarray:
        """Optimize one frame, propagating solver errors."""
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but "
                f"non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value,
            fixed_qpos,
            np.asarray(last_qpos, dtype=np.float32),
        )
        self.opt.set_min_objective(objective_fn)
        qpos = self.opt.optimize(last_qpos)
        return np.asarray(qpos, dtype=np.float32)


def build_dexpilot_retargeting(
    config: RetargetingConfig,
) -> SeqRetargeting:
    """Build strict DexPilot for the validated fixed XHand asset and SDK joint order."""
    robot_urdf = urdf.URDF.load(
        config.urdf_path,
        add_dummy_free_joints=config.add_dummy_free_joint,
        build_scene_graph=False,
    )
    urdf_name = config.urdf_path.split(os.path.sep)[-1]
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    temp_path = f"{temp_dir}/{urdf_name}"
    robot_urdf.write_xml_file(temp_path)

    robot = RobotWrapper(temp_path)

    if config.add_dummy_free_joint:
        joint_names = DUMMY_JOINT_NAMES + config.target_joint_names
    else:
        joint_names = config.target_joint_names

    optimizer = StrictDexPilotOptimizer(
        robot,
        joint_names,
        finger_tip_link_names=config.finger_tip_link_names,
        wrist_link_name=config.wrist_link_name,
        target_link_human_indices=config.target_link_human_indices,
        scaling=config.scaling_factor,
        project_dist=config.project_dist,
        escape_dist=config.escape_dist,
    )

    retargeting = SeqRetargeting(
        optimizer,
        has_joint_limits=config.has_joint_limits,
        lp_filter=LPFilter(config.low_pass_alpha),
    )
    return retargeting


class DexPilotHandRetargeter:
    def __init__(
        self,
        dexpilot_config: Any,
    ):
        dexpilot_config.validate()
        self._pinky_scale = float(dexpilot_config.pinky_scale)
        self._pinky_palm_scale = float(dexpilot_config.pinky_palm_scale)
        import yaml

        config_path = ASSET_DIR / "retargeting" / "xhand_right_dexpilot.yml"

        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        configured_joint_names = tuple(cfg.get("target_joint_names", ()))
        if configured_joint_names != XHAND_SDK_JOINT_NAMES:
            raise ValueError(
                "DexPilot target_joint_names must exactly match the canonical XHand SDK joint order"
            )

        cfg.update(
            scaling_factor=float(dexpilot_config.scaling_factor),
            low_pass_alpha=float(dexpilot_config.low_pass_alpha),
            project_dist=float(dexpilot_config.project_dist_m),
            escape_dist=float(dexpilot_config.escape_dist_m),
        )

        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = build_dexpilot_retargeting(
            RetargetingConfig.from_dict(cfg),
        )

        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in XHAND_SDK_JOINT_NAMES]
        ).astype(int)
        self.inverse_retargeted_joint_order = np.argsort(self.retargeted_joint_order)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Apply pinky segment/baseline scales before forming origin→task vectors."""
        scaled_landmarks = adaptive_retargeting_xhand(
            hand_joint_pos,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = scaled_landmarks[task_indices, :] - scaled_landmarks[origin_indices, :]
        return ref_value

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Map operator-frame (21,3) landmarks to SDK qpos.
        Invalid geometry/roundoff returns None; unexpected optimizer errors propagate to the owner.
        """
        if landmarks is None:
            return None
        valid, reason = validate_landmarks(landmarks)
        if not valid:
            logger.warning("VR landmarks rejected (%s) — no target produced", reason)
            return None

        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, TypeError, np.linalg.LinAlgError):
            logger.warning("Coordinate transform failed — no target produced")
            return None

        ref_value = self._build_ref_value(mano_landmarks)
        try:
            qpos = self.retargeter.retarget(ref_value, fixed_qpos=np.array([]))
        except nlopt.RoundoffLimited:
            logger.warning("Retargeting roundoff limit reached — no target produced")
            return None

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos_arr = np.asarray(qpos, dtype=float)

        qpos_arr = qpos_arr[self.retargeted_joint_order]

        return qpos_arr

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer, filter, projection state, and optional warm start."""
        self.retargeter.reset()
        self.retargeter.filter.reset()
        self.retargeter.optimizer.projected[:] = False

        if initial_qpos is not None and initial_qpos.shape == HAND_JOINT_SHAPE:
            qpos = np.asarray(initial_qpos, dtype=np.float32)
            if np.all(np.isfinite(qpos)):
                # Convert SDK order to the optimizer's internal order.
                qpos_retargeter = qpos[self.inverse_retargeted_joint_order]
                idx = self.retargeter.optimizer.idx_pin2target
                self.retargeter.last_qpos = qpos_retargeter[idx]
            else:
                logger.warning("initial_qpos contains NaN/Inf — falling back to neutral seed")

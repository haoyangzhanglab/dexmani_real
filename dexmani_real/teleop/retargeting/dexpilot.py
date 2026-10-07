"""DexPilot construction preserving technical-error propagation and mimic joints.

The upstream optimizer catches RuntimeError and returns the previous target;
this device-facing boundary must propagate that failure to the I/O owner.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import nlopt
import numpy as np
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig, parse_mimic_joint
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
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


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
    """Build a DexPilot sequence retargeter that propagates solver failures.

    ``config`` must come from ``RetargetingConfig.from_dict`` or its YAML loader.
    """
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

    if config.add_dummy_free_joint and config.target_joint_names is not None:
        joint_names = DUMMY_JOINT_NAMES + config.target_joint_names
    else:
        joint_names = (
            config.target_joint_names
            if config.target_joint_names is not None
            else robot.dof_joint_names
        )

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

    if 0 <= config.low_pass_alpha <= 1:
        lp_filter = LPFilter(config.low_pass_alpha)
    else:
        lp_filter = None

    has_mimic_joints, source_names, mimic_names, multipliers, offsets = parse_mimic_joint(
        robot_urdf
    )
    if has_mimic_joints and not config.ignore_mimic_joint:
        adaptor = MimicJointKinematicAdaptor(
            robot,
            target_joint_names=joint_names,
            source_joint_names=source_names,
            mimic_joint_names=mimic_names,
            multipliers=multipliers,
            offsets=offsets,
        )
        optimizer.set_kinematic_adaptor(adaptor)
        logger.info("DexPilot mimic joint adaptor enabled")

    retargeting = SeqRetargeting(
        optimizer,
        has_joint_limits=config.has_joint_limits,
        lp_filter=lp_filter,
    )
    return retargeting


class DexPilotHandRetargeter:
    def __init__(
        self,
        dexpilot_config: Any,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
    ):
        dexpilot_config.validate()
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = (
            np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        )
        self.debug_adapters = bool(debug_adapters)
        self._dexpilot_config = dexpilot_config
        self._pinky_scale = float(dexpilot_config.pinky_scale)
        self._pinky_palm_scale = float(dexpilot_config.pinky_palm_scale)
        # Keep the public output in the canonical SDK order.
        self.sdk_joint_names = XHAND_SDK_JOINT_NAMES

        self.load_retargeter()

    def load_retargeter(self):
        import yaml

        config_path = (
            ASSET_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"
        )

        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        # The YAML cannot redefine the SDK qpos order.
        configured_joint_names = tuple(cfg.get("target_joint_names", ()))
        if configured_joint_names != XHAND_SDK_JOINT_NAMES:
            raise ValueError(
                "DexPilot target_joint_names must exactly match the canonical XHand SDK joint order"
            )

        cfg.update(
            scaling_factor=float(self._dexpilot_config.scaling_factor),
            low_pass_alpha=float(self._dexpilot_config.low_pass_alpha),
            project_dist=float(self._dexpilot_config.project_dist_m),
            escape_dist=float(self._dexpilot_config.escape_dist_m),
        )

        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = build_dexpilot_retargeting(
            RetargetingConfig.from_dict(cfg),
        )

        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.sdk_joint_names]
        ).astype(int)
        self.inverse_retargeted_joint_order = np.argsort(self.retargeted_joint_order)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Build reference value from hand landmarks for retargeting.

        Applies adaptive_retargeting_xhand (pinky chain scaling) before
        computing origin→task difference vectors.
        """
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
        """Retarget raw VR landmarks (operator-frame, 21x3) to XHand joint qpos.

        Handles coordinate transform (operator → MANO), input validation,
        and NLP optimization. Expected invalid input or solver roundoff returns
        ``None``; unexpected optimizer errors propagate to the session owner.
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

        start_time = time.perf_counter() if self.debug_adapters else 0.0

        ref_value = self._build_ref_value(mano_landmarks)
        try:
            qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)
        except nlopt.RoundoffLimited:
            logger.warning("Retargeting roundoff limit reached — no target produced")
            return None

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos_arr = np.asarray(qpos, dtype=float)

        qpos_arr = qpos_arr[self.retargeted_joint_order]

        if self.debug_adapters:
            logger.info(
                "DexPilotHandRetargeter: retarget %.2f ms",
                1000 * (time.perf_counter() - start_time),
            )

        return qpos_arr

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer, filter, projection state, and optional warm start."""
        self.retargeter.reset()
        if self.retargeter.filter is not None:
            self.retargeter.filter.reset()
        self.retargeter.optimizer.projected[:] = False

        # Seed the optimizer with the current hardware pose when available.
        if initial_qpos is not None and initial_qpos.shape == HAND_JOINT_SHAPE:
            qpos = np.asarray(initial_qpos, dtype=np.float32)
            if np.all(np.isfinite(qpos)):
                # Convert SDK order to the optimizer's internal order.
                qpos_retargeter = qpos[self.inverse_retargeted_joint_order]
                idx = self.retargeter.optimizer.idx_pin2target
                self.retargeter.last_qpos = qpos_retargeter[idx]
            else:
                logger.warning("initial_qpos contains NaN/Inf — falling back to neutral seed")

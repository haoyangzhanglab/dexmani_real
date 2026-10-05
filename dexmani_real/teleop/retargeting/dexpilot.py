"""DexPilot construction preserving technical-error propagation and mimic joints.

The upstream optimizer catches RuntimeError and returns the previous target;
this device-facing boundary must propagate that failure to the I/O owner.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig, parse_mimic_joint
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.yourdfpy import DUMMY_JOINT_NAMES

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

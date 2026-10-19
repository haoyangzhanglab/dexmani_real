"""DexPilot propagates technical solver failures instead of the upstream previous-target fallback."""

from __future__ import annotations

import logging

import nlopt
import numpy as np
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting

from dexmani_real.config.control import DexPilotRetargetingParams
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.retargeter import prepare_landmarks

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


class DexPilotHandRetargeter:
    def __init__(
        self,
        dexpilot_config: DexPilotRetargetingParams,
    ):
        dexpilot_config.validate()
        self._pinky_scale = float(dexpilot_config.pinky_scale)
        self._pinky_palm_scale = float(dexpilot_config.pinky_palm_scale)
        robot = RobotWrapper(str(XHAND_RIGHT_URDF_PATH))
        optimizer = StrictDexPilotOptimizer(
            robot,
            list(XHAND_SDK_JOINT_NAMES),
            finger_tip_link_names=list(XHAND_FINGERTIP_LINK_NAMES),
            wrist_link_name="right_hand_link",
            scaling=dexpilot_config.scaling_factor,
            project_dist=dexpilot_config.project_dist_m,
            escape_dist=dexpilot_config.escape_dist_m,
        )
        self.retargeter = SeqRetargeting(
            optimizer,
            has_joint_limits=True,
            lp_filter=LPFilter(dexpilot_config.low_pass_alpha),
        )

        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in XHAND_SDK_JOINT_NAMES]
        ).astype(int)
        self.inverse_retargeted_joint_order = np.argsort(self.retargeted_joint_order)

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Map (21,3) landmarks to SDK qpos; invalid geometry/roundoff returns None, other errors propagate."""
        mano = prepare_landmarks(
            landmarks,
            pinky_scale=self._pinky_scale,
            pinky_palm_scale=self._pinky_palm_scale,
        )
        if mano is None:
            return None
        ref_value = mano[self.indices[1]] - mano[self.indices[0]]
        try:
            qpos = self.retargeter.retarget(ref_value, fixed_qpos=np.array([]))
        except nlopt.RoundoffLimited:
            logger.warning("Retargeting roundoff limit reached — no target produced")
            return None

        return qpos[self.retargeted_joint_order]

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

"""DexPilot retargeting with a human-flexion prior (in-repo subclass wrapper).

dex-retargeting 0.4.6 dropped the original DexPilot paper's open-hand
regularizer ``γ·‖q‖²`` (the ``# gamma=2.5e-3`` parameter is commented out in
``DexPilotOptimizer.__init__``) and keeps only a *gradient-only* temporal term
``norm_delta·‖x − last_qpos‖²``.  That anchors the under-determined null-space
(MCP/PIP distribution, thumb bend/rota distribution) to the previous frame, so
it never collapses onto a natural-looking hand.

This module restores a prior without touching site-packages: it subclasses
``DexPilotOptimizer`` and wraps the per-frame objective so a prior
``γ·Σ mask·(x − q_ref)²`` is added to **both** the scalar (so SLSQP's line
search sees it) and the gradient.  ``q_ref`` is the *per-frame* human flexion
reference (set via ``set_prior_reference`` before each ``retarget``), masked to
the 10 flexion joints — this matches the robot's flexion to the operator's
actual per-frame hand shape rather than a fixed pose.  At ``prior_weight == 0``
the behavior is bit-identical to the vanilla optimizer.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

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


class PriorDexPilotOptimizer(DexPilotOptimizer):
    """DexPilotOptimizer plus a masked quadratic human-flexion prior.

    ``x`` is the NLopt optimization variable in target-joint order (== SDK order
    for the XHand).  ``prior_mask`` and the per-frame ``q_prior`` reference must
    have the same length/order.
    """

    def __init__(
        self, *args: Any, prior_weight: float = 0.0, prior_mask: np.ndarray | None = None, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.prior_weight = float(prior_weight)
        if prior_mask is None:
            self.prior_mask = None
        else:
            self.prior_mask = np.asarray(prior_mask, dtype=np.float64)
        self._prior_reference: np.ndarray | None = None  # per-frame (target-order)

    def set_prior_reference(self, q_prior: np.ndarray) -> None:
        """Set the current frame's human-flexion reference (target/SDK order)."""
        self._prior_reference = np.asarray(q_prior, dtype=np.float64)

    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        base = super().get_objective_function(target_vector, fixed_qpos, last_qpos)
        ref = self._prior_reference
        if ref is None or self.prior_mask is None or self.prior_weight <= 0:
            return base
        weighted_mask = self.prior_weight * self.prior_mask

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            result = base(x, grad)
            diff = x - ref
            # Add the prior to BOTH value and gradient — unlike the temporal
            # term upstream (gradient-only), so SLSQP's line search sees it.
            if grad.size > 0:
                grad += 2.0 * weighted_mask * diff
            return result + float(np.sum(weighted_mask * diff * diff))

        return objective


def build_dexpilot_retargeting(
    config: RetargetingConfig,
    *,
    prior_weight: float,
    prior_mask: np.ndarray | None,
) -> SeqRetargeting:
    """Build a DexPilot ``SeqRetargeting`` with the human-flexion prior.

    Mirrors the ``dexpilot`` branch of ``RetargetingConfig.build()``
    (site-packages ``retargeting_config.py``), but constructs
    ``PriorDexPilotOptimizer`` instead of ``DexPilotOptimizer``.  ``config``
    must be a validated ``RetargetingConfig`` (from ``from_dict`` / the YAML).

    The XHand URDF has no ``<mimic>`` joints, so the mimic-adaptor branch is
    normally dead here; it is retained for parity with upstream in case the URDF
    ever gains one.
    """
    robot_urdf = urdf.URDF.load(
        config.urdf_path, add_dummy_free_joints=config.add_dummy_free_joint, build_scene_graph=False
    )
    urdf_name = config.urdf_path.split(os.path.sep)[-1]
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    temp_path = f"{temp_dir}/{urdf_name}"
    robot_urdf.write_xml_file(temp_path)

    robot = RobotWrapper(temp_path)

    if config.add_dummy_free_joint and config.target_joint_names is not None:
        joint_names = DUMMY_JOINT_NAMES + config.target_joint_names
    else:
        joint_names = config.target_joint_names if config.target_joint_names is not None else robot.dof_joint_names

    optimizer = PriorDexPilotOptimizer(
        robot,
        joint_names,
        finger_tip_link_names=config.finger_tip_link_names,
        wrist_link_name=config.wrist_link_name,
        target_link_human_indices=config.target_link_human_indices,
        scaling=config.scaling_factor,
        project_dist=config.project_dist,
        escape_dist=config.escape_dist,
        prior_weight=prior_weight,
        prior_mask=prior_mask,
    )

    if 0 <= config.low_pass_alpha <= 1:
        lp_filter = LPFilter(config.low_pass_alpha)
    else:
        lp_filter = None

    has_mimic_joints, source_names, mimic_names, multipliers, offsets = parse_mimic_joint(robot_urdf)
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
        logger.info("DexPilot mimic joint adaptor enabled (prior path)")

    retargeting = SeqRetargeting(
        optimizer,
        has_joint_limits=config.has_joint_limits,
        lp_filter=lp_filter,
    )
    return retargeting

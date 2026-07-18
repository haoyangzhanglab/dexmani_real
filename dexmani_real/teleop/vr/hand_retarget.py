"""VR-to-XHand retargeting via dex_retargeting + adaptive finger scaling.

Two landmark-space adaptations compensate for human-robot kinematic mismatches
before reference vectors are computed:

1. adaptive_retargeting_thumb — scales thumb_tip to match XHand mechanical thumb
   length (~23% longer than MANO model).  Adaptive: more when extended, less
   when curled — drives rot2 (IP flexion) toward zero in neutral poses while
   preserving curl range for opposition/grip gestures.

2. adaptive_retargeting_xhand (LeFranX) — scales pinky chain (MCP→PIP→DIP→TIP)
   based on extension state, compensating for human-robot finger length
   differences.  Uniform scaling along the kinematic chain.

Both operate on MANO-space landmarks and modify disjoint landmark indices
(thumb: 4, pinky: 18-20), so order is irrelevant.

Refs:
  LeFranX vr_hand_detector_adapter.py:27-84 (pinky)
  CL-20260701 thumb FK analysis: robot wrist→thumb_tip 0.161m vs MANO 0.131m
"""

from __future__ import annotations

__all__ = ["XHandRetargeter", "adaptive_retargeting_thumb", "adaptive_retargeting_xhand"]

import os
import tempfile
import time
import numpy as np
import torch
import yaml
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig, parse_mimic_joint
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ── Pinky landmark indices (MediaPipe convention) ──
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

# Pinky adaptive scaling — calibrated to actual MANO-space pinky
# MCP→TIP distances observed across 5 teleop recordings (500+ frames).
# LeFranX original values (_MIN=0.03, _MAX=0.10) were tuned for a
# different MANO space where the pinky reaches 0.10 m at full extension.
# Our MANO space maxes out at ~0.073 m, so we calibrate the range to match.
#
# Parameters selected via offline grid search on real recording data
# (20260701_161732, 3431 frames): sweep of max_scale ∈ [2.0, 2.2, 2.4, 2.6]
# with calibrated _MIN/_MAX.  Chose 2.4 as the best balance between
# straightening the extended pinky and preserving curl range.

_PINKY_MIN_EXTENSION = 0.0280  # P2 of MANO pinky MCP→TIP (fully curled)
_PINKY_MAX_EXTENSION = 0.074  # P98 of MANO pinky MCP→TIP (fully extended)
_PINKY_BASE_SCALE = 1.15  # minimum scaling for curled: enough push so J11 can reach
_PINKY_MAX_SCALE = 2.40  # maximum scaling: match robot pinky length when extended


def adaptive_retargeting_xhand(landmarks: np.ndarray) -> np.ndarray:
    """Apply adaptive pinky scaling for XHand robot (LeFranX approach).

    Compensates for human-to-robot finger length differences by adaptively
    scaling pinky chain segments based on finger extension state.
    Scales more when extended (for reaching), less when curled (for fist-making).

    Operates on MANO-space landmarks — modifies pinky PIP/DIP/TIP positions
    in-place on a copy. The scaled landmarks are then used directly by the
    existing retargeting pipeline.

    Ref: LeFranX vr_hand_detector_adapter.py:27-84

    Args:
        landmarks: (21, 3) array in MANO coordinate space.

    Returns:
        (21, 3) array with pinky chain scaled (new copy, input unchanged).
    """
    landmarks = landmarks.copy()

    # Finger extension: distance from MCP to TIP
    pinky_extension = float(np.linalg.norm(landmarks[_PINKY_TIP] - landmarks[_PINKY_MCP]))

    # Normalize extension ratio (0.0 = fully curled, 1.0 = fully extended)
    extension_ratio = np.clip(
        (pinky_extension - _PINKY_MIN_EXTENSION) / (_PINKY_MAX_EXTENSION - _PINKY_MIN_EXTENSION),
        0.0,
        1.0,
    )

    # Adaptive scaling: more when extended, less when curled
    adaptive_scale = _PINKY_BASE_SCALE + (_PINKY_MAX_SCALE - _PINKY_BASE_SCALE) * extension_ratio

    # Apply progressive scaling along the kinematic chain (MCP→PIP→DIP→TIP).
    # Each segment is extended from the (possibly modified) parent joint.
    mcp_to_pip = landmarks[_PINKY_PIP] - landmarks[_PINKY_MCP]
    landmarks[_PINKY_PIP] = landmarks[_PINKY_MCP] + mcp_to_pip * adaptive_scale

    pip_to_dip = landmarks[_PINKY_DIP] - landmarks[_PINKY_PIP]
    landmarks[_PINKY_DIP] = landmarks[_PINKY_PIP] + pip_to_dip * adaptive_scale

    dip_to_tip = landmarks[_PINKY_TIP] - landmarks[_PINKY_DIP]
    landmarks[_PINKY_TIP] = landmarks[_PINKY_DIP] + dip_to_tip * adaptive_scale

    return landmarks


# ── Thumb adaptive scaling parameters ──
# XHand thumb mechanical length (wrist→thumb_tip) is ~0.161 m at neutral
# (bend=0, rot1=0, rot2=0).  The MANO model averages ~0.131 m — a ~23 %
# mismatch.  Without compensation the DexPilot optimizer uses rot2 (thumb IP
# flexion) purely to shorten the kinematic chain, keeping it at ~1.1–1.3 rad
# (63–75°) regardless of the human thumb state.
#
# Adaptive scaling: when the human thumb is extended (long wrist→tip),
# scale the target up toward the robotʼs mechanical length so rot2 can drop
# toward zero.  When the human thumb is curled, scale conservatively to
# preserve the optimizerʼs ability to increase rot2 for opposition / grip.

_THUMB_MIN_EXTENSION = 0.105  # P3  of human wrist→thumb_tip (curled)
_THUMB_MAX_EXTENSION = 0.140  # P97 of human wrist→thumb_tip (extended)
_THUMB_BASE_SCALE = 1.02  # minimum scaling: slight push for baseline J2 engagement
_THUMB_MAX_SCALE = 1.35  # maximum scaling: balanced approach to robot 0.161m length


def adaptive_retargeting_thumb(landmarks: np.ndarray) -> np.ndarray:
    """Scale thumb tip to match XHand mechanical thumb length.

    The DexPilot reference vectors only use thumb_tip (landmark 4) — not
    the intermediate thumb joints (CMC=1, MCP=2, IP=3).  Scaling just the
    tip is sufficient to shift the wrist→thumb_tip distance into the
    robotʼs achievable range at low rot2.

    Operates on MANO-space landmarks.  The pinky scaling
    (adaptive_retargeting_xhand) operates on landmarks 18–20, so the two
    are independent and can be composed in any order.

    Args:
        landmarks: (21, 3) array in MANO coordinate space.

    Returns:
        (21, 3) array with thumb_tip scaled (new copy, input unchanged).
    """
    landmarks = landmarks.copy()

    wrist = landmarks[0]
    thumb_tip = landmarks[4]

    extension = float(np.linalg.norm(thumb_tip - wrist))
    extension_ratio = np.clip(
        (extension - _THUMB_MIN_EXTENSION) / (_THUMB_MAX_EXTENSION - _THUMB_MIN_EXTENSION),
        0.0,
        1.0,
    )
    scale = _THUMB_BASE_SCALE + (_THUMB_MAX_SCALE - _THUMB_BASE_SCALE) * extension_ratio
    landmarks[4] = wrist + (thumb_tip - wrist) * scale

    return landmarks


class XHandDexPilotOptimizer(DexPilotOptimizer):
    """DexPilot variant with balanced wrist→fingertip vs between-finger weights.

    The original DexPilotOptimizer assigns weight ~15 to wrist→fingertip vectors
    and weight 1 to between-finger vectors, causing distal joints (J2, J11) to
    serve as binary length compensators instead of tracking finger flexion.

    This subclass rebalances the weights so between-finger vectors (which carry
    the true finger flexion signal) have more influence on the optimizer.
    """

    def __init__(self, wrist_weight: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.wrist_weight = float(wrist_weight)

    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        """Same as parent but with configurable wrist→fingertip weight.

        The only change from DexPilotOptimizer.get_objective_function() is the
        weight computation: wrist→fingertip vectors use self.wrist_weight * n_fingers
        instead of hardcoded len_proj + n_fingers (~15).
        """
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # ── KEY CHANGE: balanced wrist→fingertip weight ──
        # Original: wrist weight = len_proj + n_fingers ≈ 15 (dominates between-finger weight 1)
        # New:       wrist weight = wrist_weight * n_fingers → DEFAULT 10 (= 2.0 * 5)
        #            This gives between-finger vectors a stronger voice in the optimizer,
        #            reducing the optimizer's reliance on distal joints for length compensation.
        wrist_finger_weight = self.wrist_weight * self.num_fingers
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * wrist_finger_weight])
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class XHandRetargeter:
    def __init__(
        self,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
        low_pass_alpha: float | None = None,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        self.debug_adapters = bool(debug_adapters)
        self.last_debug = {}

        self.sapien_joint_names = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]

        self.load_retargeter()

        # Override the YAML low_pass_alpha (tuned at 50Hz) — callers running at a
        # different control rate pass a tau-preserving value (utils.signal_utils).
        if low_pass_alpha is not None:
            self.low_pass_alpha = low_pass_alpha

    def load_retargeter(self):
        config_path = ASSET_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"

        # Load YAML config
        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        # Extract XHand-specific wrist_weight (default 2.0 if not specified)
        wrist_weight = float(cfg.get("wrist_weight", 2.0))
        scaling_factor = float(cfg.get("scaling_factor", 1.0))

        # ── Manual build (replaces RetargetingConfig.build()) ──
        # We build the retargeter manually so we can inject XHandDexPilotOptimizer
        # with balanced wrist→fingertip weights.
        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        urdf_path = str(ASSET_DIR / "robots" / cfg["urdf_path"])

        robot_urdf = urdf.URDF.load(urdf_path, add_dummy_free_joints=False, build_scene_graph=False)
        urdf_name = os.path.basename(urdf_path)
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = os.path.join(temp_dir, urdf_name)
        robot_urdf.write_xml_file(temp_path)
        robot = RobotWrapper(temp_path)

        joint_names = robot.dof_joint_names
        target_joint_names = cfg.get("target_joint_names", joint_names)

        optimizer = XHandDexPilotOptimizer(
            robot=robot,
            target_joint_names=target_joint_names,
            finger_tip_link_names=cfg["finger_tip_link_names"],
            wrist_link_name=cfg["wrist_link_name"],
            target_link_human_indices=cfg.get("target_link_human_indices"),
            scaling=scaling_factor,
            project_dist=cfg.get("project_dist", 0.03),
            escape_dist=cfg.get("escape_dist", 0.05),
            wrist_weight=wrist_weight,
        )

        # Set up mimic joints (same as RetargetingConfig.build())
        has_mimic, src_names, mimic_names, multipliers, offsets = parse_mimic_joint(robot_urdf)
        if has_mimic:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=target_joint_names,
                source_joint_names=src_names,
                mimic_joint_names=mimic_names,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)

        # Low-pass filter
        alpha = float(cfg.get("low_pass_alpha", 0.6))
        lp_filter = LPFilter(alpha) if 0 <= alpha <= 1 else None

        self.retargeter = SeqRetargeting(optimizer, has_joint_limits=True, lp_filter=lp_filter)
        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.sapien_joint_names]
        ).astype(int)

    @property
    def low_pass_alpha(self) -> float:
        """Current LPFilter alpha (new-value weight: 1.0 = pass-through, →0 = freeze)."""
        return float(self.retargeter.filter.alpha)

    @low_pass_alpha.setter
    def low_pass_alpha(self, value: float) -> None:
        """Tune the built-in EMA smoothing strength at runtime.

        dex_retargeting applies LPFilter after NLP optimisation, before returning
        qpos: ``y += alpha * (x - y)`` — alpha is the NEW-value weight.
        Default from YAML config: 0.6 (60 % new + 40 % old per frame at 50 Hz).

        1.0 = pass-through (raw optimizer output, no smoothing)
        0.6 = default (moderate smoothing)
        →0  = ever-stronger smoothing; 0.0 freezes the output entirely
        """
        self.retargeter.filter.alpha = float(value)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Build reference value from hand landmarks for retargeting.

        Applies two landmark-space adaptations before computing origin→task
        difference vectors:

        1. adaptive_retargeting_thumb — scale thumb_tip to match robot
           mechanical length, enabling low rot2 in neutral poses.
        2. adaptive_retargeting_xhand (LeFranX) — scale pinky chain for
           human-robot finger length mismatch.
        """
        if self.retargeting_type == "position":
            return hand_joint_pos[self.indices, :]

        # Scale thumb tip to compensate for XHand's ~23% longer mechanical
        # thumb — drives rot2 toward zero when the human thumb is extended
        # while preserving curl range for opposition / grip.
        scaled_landmarks = adaptive_retargeting_thumb(hand_joint_pos)
        # LeFranX: scale pinky chain on landmarks before computing ref vectors.
        # This directly modifies PIP/DIP/TIP positions along the MCP→TIP chain,
        # compensating for human-robot finger length differences.
        scaled_landmarks = adaptive_retargeting_xhand(scaled_landmarks)

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = scaled_landmarks[task_indices, :] - scaled_landmarks[origin_indices, :]
        return ref_value

    def retarget(self, hand_joint_pos: np.ndarray | None) -> np.ndarray | None:
        if hand_joint_pos is None:
            return None

        start_time = time.time()

        ref_value = self._build_ref_value(hand_joint_pos)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos = np.asarray(qpos, dtype=float)[self.retargeted_joint_order]

        if self.debug_adapters:
            self.last_debug = {
                "retarget_ms": 1000 * (time.time() - start_time),
                "adaptives": "thumb(scale_thumb_tip) + pinky(LeFranX chain scaling)",
            }
            logger.info("retarget_debug: %s", self.last_debug)

        return qpos

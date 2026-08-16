"""Shared constants for the planning module.

Constants that are used by multiple planning submodules live here
to avoid duplication and ensure a single source of truth.
"""

from __future__ import annotations

from dexmani_real.utils.schema import XHAND_SDK_JOINT_NAMES

# ═════════════════════════════════════════════════════════════════════════════
# Hand joint order remap: XHand SDK → URDF / Pinocchio
# ═════════════════════════════════════════════════════════════════════════════
#
# Canonical SDK order (utils.schema.XHAND_SDK_JOINT_NAMES):
#   [thumb_bend, thumb_rota1, thumb_rota2,
#    index_bend, index_j1,   index_j2,
#    mid_j1,    mid_j2,
#    ring_j1,   ring_j2,
#    pinky_j1,  pinky_j2]
#
# URDF / Pinocchio model.names (alphabetical by joint name):
#   [index_bend, index_j1,   index_j2,
#    mid_j1,    mid_j2,
#    pinky_j1,  pinky_j2,
#    ring_j1,   ring_j2,
#    thumb_bend, thumb_rota1, thumb_rota2]
#
# map[i] = SDK index for URDF / Pinocchio slot i. Deriving by name keeps the
# planning permutation coupled to the cross-process qpos contract instead of a
# second unexplained index literal.
# Used by: collision_model.CollisionModel, hand_kinematics.HandKinematics
_HAND_URDF_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
)
HAND_SDK_TO_URDF_IDX: tuple[int, ...] = tuple(
    XHAND_SDK_JOINT_NAMES.index(name) for name in _HAND_URDF_JOINT_NAMES
)

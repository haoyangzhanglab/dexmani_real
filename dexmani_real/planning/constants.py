"""Shared constants for the planning module.

Constants that are used by multiple planning submodules live here
to avoid duplication and ensure a single source of truth.
"""

from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# Hand joint order remap: XHand SDK → URDF / Pinocchio
# ═════════════════════════════════════════════════════════════════════════════
#
# SDK (finger-grouped, xhand.py JOINT_NAMES):
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
# map[i] = SDK index for URDF / Pinocchio slot i.
# Used by: collision_model.CollisionModel, hand_kinematics.HandKinematics
HAND_SDK_TO_URDF_IDX: tuple[int, ...] = (3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2)

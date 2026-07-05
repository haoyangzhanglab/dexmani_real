"""Unified collision configuration — single source of truth for all safety parameters.

Replaces the previously scattered DESK_SAFE_Z / HAND_SAFE_MARGIN /
HAND_EXTENSION_BELOW_EEF constants that were spread across 4 files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from dexmani_real.utils.serialization import FromDictMixin


@dataclass
class CollisionConfig(FromDictMixin):
    """Unified collision detection and safety margin configuration.

    All parameters in meters / world frame unless noted otherwise.

    Two complementary safety layers:
      - CollisionModel FCL: box obstacle registered via add_table(), checked by
        teleop env gate (ik.py) and path validation (planner.py).  Uses actual
        hand joint angles in 19-DOF mode.
      - FingertipDeskSafety FK: Pinocchio FK computes five fingertip world Z
        coordinates and compares min against table surface + safe margin.
        Fast, zero-obstacle, independent layer.
    """

    # ── Table geometry (world frame, shared by CollisionModel.add_table + FingertipDeskSafety) ──
    table_z_world: float = 0.0
    """Table surface height in world frame (meters)."""

    table_x_center: float = 0.5
    """Table box centre X in robot base frame (meters)."""

    table_half_x: float = 1.0
    """Table box half-extent X (meters)."""

    table_half_y: float = 2.0
    """Table box half-extent Y (meters)."""

    table_half_z: float = 0.04
    """Table box half-extent Z (meters).  Box top = table_z_world, bottom = table_z_world - 2*half_z."""

    # ── Hand geometry (derived from collision URDF, home hand pose) ──
    hand_extension_below_eef: float = 0.076
    """Distance from EEF to lowest fingertip (pinky_tip) at home hand pose (m).
    FK-measured from xarm7_xhand_collision.urdf."""

    hand_safe_margin: float = 0.03
    """Minimum fingertip-to-table clearance (meters)."""

    # ── Fingertip link identification (from collision URDF) ──
    # xarm7_xhand_collision.urdf link order (0-indexed):
    #   0-11: arm (link_base..custom_eef_link)
    #   12-41: hand (right_hand_link..pinky_tip)
    #   20=thumb_rota_tip, 26=index_rota_tip, 31=mid_tip, 36=ring_tip, 41=pinky_tip
    fingertip_link_ids: tuple[int, ...] = (20, 26, 31, 36, 41)
    fingertip_link_names: tuple[str, ...] = (
        "thumb_tip", "index_tip", "mid_tip", "ring_tip", "pinky_tip",
    )

    # ── Environment collision ──
    enable_env_collision: bool = True
    """Enable environment (table/obstacle) collision detection.

    When True, registers an FCL box obstacle in CollisionModel (teleop env gate
    + path validation) and activates FK fingertip Z desk safety checks.  Set to
    False to disable all environment collision layers.
    """

    collision_step_size: float = 0.02
    """Joint-space step size [rad] for dense segment collision interpolation.
    Default 0.02 rad ~1.15° (ref: dimos collision_step_size)."""

    desk_safety_step_rad: float = 0.05
    """Joint-space step size [rad] for fingertip desk safety path checks.

    Default 0.05 rad (≈2.9°).  The desk is a continuous plane — fingertip Z
    changes smoothly, so a coarser step than collision_step_size (0.02) is
    sufficient and halves FK calls (P4 optimisation)."""

    # ── Tier margins for env collision Z-filter ──
    tier1_z_margin: float = 0.05
    """Tier 1 Z-min pre-filter margin [m].
    4 cm fingertip mesh half-extent + 1 cm safety margin."""

    tier2_z_margin: float = 0.25
    """Tier 2 per-geometry Z skip margin [m].
    Robot geometries whose FK centre is this far above an obstacle are skipped
    in the expensive FCL mesh-mesh check.  Only hand/wrist/forearm geometries
    pass through to Tier 2."""

    # ── Derived properties ──

    @property
    def desk_safe_z(self) -> float:
        """EEF must be above this Z to guarantee no fingertip-table collision.

        Computed as: table_z_world + hand_extension_below_eef + hand_safe_margin.

        For default values: 0.0 + 0.076 + 0.03 = 0.106 m.
        """
        return self.table_z_world + self.hand_extension_below_eef + self.hand_safe_margin

    @property
    def fingertip_threshold(self) -> float:
        """Fingertip world Z must be strictly greater than this value.
        Computed as: table_z_world + hand_safe_margin.
        """
        return self.table_z_world + self.hand_safe_margin

    # ── Factory / utility ──

    def with_overrides(self, **kwargs) -> "CollisionConfig":
        """Return a new CollisionConfig with specified fields overridden.

        Uses ``dataclasses.replace()`` (stdlib). Automatically stays in sync
        when new fields are added to the dataclass.
        """
        return replace(self, **kwargs)


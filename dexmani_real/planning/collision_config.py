"""Unified collision configuration — single source of truth for all safety parameters.

Replaces the previously scattered DESK_SAFE_Z / HAND_SAFE_MARGIN /
HAND_EXTENSION_BELOW_EEF constants that were spread across 4 files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollisionConfig:
    """Unified collision detection and safety margin configuration.

    All parameters in meters / world frame unless noted otherwise.

    Two complementary safety layers:
      - Geometric FK: Pinocchio FK computes fingertip world Z → compare to
        table surface.  Zero-cost, no MPlib point cloud needed.
      - MPlib point cloud: Dense point cloud added to MPlib octree, used by
        plan_screw/plan_qpos/IK for avoidance.  Costs IK success rate (~53%
        vs 100% without).

    Default mode is "geometric_fk" — the MPlib point cloud is only used
    when explicitly requested (e.g. for non-table obstacles).
    """

    # ── Table geometry (world frame) ──
    table_z_world: float = 0.0
    """Table surface height in world frame (meters)."""

    # ── Hand geometry (derived from collision URDF, home hand pose) ──
    hand_extension_below_eef: float = 0.076
    """Distance from EEF to lowest fingertip (pinky_tip) at home hand pose (m).
    FK-measured from xarm7_xhand_collision.urdf."""

    hand_safe_margin: float = 0.03
    """Minimum fingertip-to-table clearance (meters)."""

    # ── Table-top object interaction ──
    table_object_max_height: float = 0.0
    """Maximum height of objects on the table surface (m).
    Default 0.0 = no objects.  Set to e.g. 0.10 for a 10 cm tall box."""

    table_object_safe_margin: float = 0.02
    """Additional EEF-to-object-top clearance when objects are present (m)."""

    # ── Fingertip link identification (from collision URDF) ──
    # xarm7_xhand_collision.urdf link order (0-indexed):
    #   0-11: arm (link_base..custom_eef_link)
    #   12-41: hand (right_hand_link..pinky_tip)
    #   20=thumb_rota_tip, 26=index_rota_tip, 31=mid_tip, 36=ring_tip, 41=pinky_tip
    fingertip_link_ids: tuple[int, ...] = (20, 26, 31, 36, 41)
    fingertip_link_names: tuple[str, ...] = (
        "thumb_tip", "index_tip", "mid_tip", "ring_tip", "pinky_tip",
    )

    # ── Environment collision mode ──
    env_collision_mode: str = "geometric_fk"
    """Collision detection strategy for environment (table/objects).

    Options:
      - "geometric_fk": Pinocchio FK fingertip Z detection (fast, accurate, default).
      - "mplib_pointcloud": MPlib octree via add_point_cloud() (costs IK success rate).
      - "none": No environment collision detection.
    """

    # ── Pre-filter ──
    reject_below_desk_z: bool = True
    """When True, reject planning targets whose EEF Z is below desk_safe_z.
    This is a fast pre-filter; actual collision detection uses fingertip FK.
    Disable for desk-interaction tests (--test-desk, --with-objects)."""

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

    @property
    def eef_safe_z_with_objects(self) -> float:
        """EEF safe Z considering table-top objects.

        Computed as: table_z_world + table_object_max_height
                    + table_object_safe_margin
                    + hand_extension_below_eef + hand_safe_margin.
        """
        return (
            self.table_z_world
            + self.table_object_max_height
            + self.table_object_safe_margin
            + self.hand_extension_below_eef
            + self.hand_safe_margin
        )

    # ── Factory / utility ──

    def with_overrides(self, **kwargs) -> "CollisionConfig":
        """Return a new CollisionConfig with specified fields overridden."""
        new_dict = {
            k: getattr(self, k)
            for k in [
                "table_z_world", "hand_extension_below_eef", "hand_safe_margin",
                "table_object_max_height", "table_object_safe_margin",
                "fingertip_link_ids", "fingertip_link_names",
                "env_collision_mode", "reject_below_desk_z",
            ]
        }
        new_dict.update(kwargs)
        return CollisionConfig(**new_dict)

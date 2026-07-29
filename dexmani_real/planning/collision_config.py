"""Unified collision configuration — FCL table obstacle, tier margins."""

from dataclasses import dataclass


@dataclass
class CollisionConfig:
    """Unified collision detection and safety margin configuration.

    All parameters in meters / world frame unless noted otherwise.

    Environment collision uses CollisionModel FCL: box obstacle registered via
    add_table(), checked by teleop env gate (ik.py) and path validation
    (planner.py).  Uses actual hand joint angles in 19-DOF mode.
    """

    # ── Table geometry (world frame) ──
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

    # ── Environment collision ──
    enable_env_collision: bool = True
    """Enable environment (table/obstacle) collision detection via FCL.

    When True, registers an FCL box obstacle in CollisionModel (teleop env gate
    + path validation).  Set to False to disable all environment collision layers.
    """

    # ── Tier margins for env collision Z-filter ──
    tier1_z_margin: float = 0.05
    """Tier 1 Z-min pre-filter margin [m].
    4 cm fingertip mesh half-extent + 1 cm safety margin."""

    tier2_z_margin: float = 0.25
    """Tier 2 per-geometry Z skip margin [m].
    Robot geometries whose FK centre is this far above an obstacle are skipped
    in the expensive FCL mesh-mesh check.  Only hand/wrist/forearm geometries
    pass through to Tier 2."""

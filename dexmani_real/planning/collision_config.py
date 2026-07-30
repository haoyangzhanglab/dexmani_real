"""Collision configuration — kept for backward compatibility.

Environment collision detection (FCL-based) has been removed.
Self-collision (Pinocchio-based) remains active and does not require
this configuration.
"""

from dataclasses import dataclass


@dataclass
class CollisionConfig:
    """Backward-compatible no-op collision configuration.

    Previously held table geometry and FCL tier margin settings.
    All fields have been removed — the dataclass exists only so
    existing ``CollisionConfig()`` instantiation sites continue
    to work without modification.
    """

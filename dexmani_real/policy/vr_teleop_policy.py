"""Compatibility imports for the VR teleoperation loop.

The implementation lives in :mod:`dexmani_real.teleop`, where mapping,
sampling, and safety code can be read as robotics concepts.
"""

from __future__ import annotations

from dexmani_real.teleop.config import TeleopConfig as PolicyConfig
from dexmani_real.teleop.loop import teleop_loop as policy_loop

__all__ = ["PolicyConfig", "policy_loop"]

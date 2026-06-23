"""Teleop module — VR teleoperation control.

Split into core / vr / control sub-packages:
  - core:    TeleopController
  - vr:      ArmMapper, HandRetargeter, VRTracker
  - control: KeyboardHandler, safety checks

Top-level __init__.py provides backward-compatible re-exports.
"""

# ── Controller ──
from dexmani_real.teleop.core.controller import ControllerState, TeleopController
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.control import safety

# ── VR tracking ──
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.dummy_tracker import DummyTracker
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
from dexmani_real.teleop.vr.ref_adapter import XHandRefAdapter
from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker

__all__ = [
    # ── Controller ──
    "ControllerState",
    "ControlSignal",
    "KeyboardHandler",
    "safety",
    "TeleopController",
    # ── VR ──
    "ArmWristMapper",
    "QuestHandTracker",
    "XHandRefAdapter",
    "XHandRetargeter",
]

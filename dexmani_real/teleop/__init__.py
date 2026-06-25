"""Teleop module — VR teleoperation control.

Split into core / vr / control sub-packages:
  - core:    TeleopController
  - vr:      ArmMapper, HandRetargeter, VRTracker
  - control: KeyboardHandler
"""

from dexmani_real.teleop.core.controller import ControllerState, TeleopController
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
from dexmani_real.teleop.vr.dummy_tracker import DummyTracker
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter, adaptive_retargeting_xhand
from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker

__all__ = [
    "ControllerState",
    "ControlSignal",
    "KeyboardHandler",
    "TeleopController",
    "TeleopPipeline",
    "ArmWristMapper",
    "QuestHandTracker",
    "XHandRetargeter",
    "adaptive_retargeting_xhand",
]

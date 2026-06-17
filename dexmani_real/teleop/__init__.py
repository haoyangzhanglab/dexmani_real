"""Teleop 模块 — VR 遥操作控制。

拆分为 core / vr / control 三个子包：
  - core:    TeleopController, ErrorHandler, TrackingQuality
  - vr:      ArmMapper, HandRetargeter, VRTracker
  - control: KeyboardHandler, SafetyChecker

顶层 __init__.py 提供向后兼容重导出。
"""

# ── 控制器 ──
from dexmani_real.teleop.core.controller import ControllerState, TeleopController
from dexmani_real.teleop.core.error_handler import TeleopErrorHandler
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.control.safety import SafetyChecker
from dexmani_real.teleop.core.tracking import TrackingQuality, TrackingQualityConfig, TrackingQualityResult

# ── VR 追踪 ──
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
    "SafetyChecker",
    "TeleopController",
    "TeleopErrorHandler",
    "TrackingQuality",
    "TrackingQualityConfig",
    "TrackingQualityResult",
    # ── VR ──
    "ArmWristMapper",
    "DummyTracker",
    "QuestHandTracker",
    "XHandRefAdapter",
    "XHandRetargeter",
]

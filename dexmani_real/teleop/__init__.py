"""Teleop 模块 — VR 遥操作控制。

合并了原 controller/ 和 teleop/ 两个包，参考 LeFranX 的 teleoperators/ 模式。
"""

# ── 控制器（原 controller/，现合并到 teleop/ 子文件）──
from dexmani_real.teleop.controller import ControllerState, TeleopController
from dexmani_real.teleop.error_handler import TeleopErrorHandler
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.safety import SafetyChecker
from dexmani_real.teleop.tracking import TrackingQuality, TrackingQualityConfig, TrackingQualityResult

# ── VR 追踪（原 teleop/ 子文件）──
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.dummy_tracker import DummyTracker
from dexmani_real.teleop.hand_retarget import XHandRetargeter
from dexmani_real.teleop.ref_adapter import XHandRefAdapter
from dexmani_real.teleop.visualizer import QuestHandVisualizer
from dexmani_real.teleop.vr_tracker import QuestHandTracker

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
    "QuestHandVisualizer",
    "XHandRefAdapter",
    "XHandRetargeter",
]

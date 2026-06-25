"""Pre-Flight check — hardware readiness verification before entering teleop.

Shared by keyboard_teleop_real.py and any other entry point that needs
a pre-operation safety checklist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class PreFlightReport:
    passed: bool
    checks: list[tuple[str, bool | None, str]]  # (name, ok, detail)


def preflight_check(robot) -> PreFlightReport:
    """Pre-Flight hardware readiness checklist.

    Checks:
      1. Arm connected
      2. Arm error-free
      3. Joint angles valid (non-NaN, within limits)
      4. Hand connected (degraded — warning only)
      5. Hand communication ok (degraded — warning only)

    Returns PreFlightReport with per-check pass/fail/warn.
    """
    checks: list[tuple[str, bool | None, str]] = []

    # ── Arm connection ──
    arm_connected = robot.arm.is_connected()
    checks.append(("arm 连接", arm_connected, "" if arm_connected else "arm 未连接"))

    # ── Arm error free ──
    arm_ok = not robot.arm.is_error()
    checks.append(("arm 无错误", arm_ok, "" if arm_ok else robot.arm.last_error_message))

    # ── Joint angles valid ──
    state = robot.arm.get_state()
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    has_nan = not np.all(np.isfinite(qpos))
    checks.append(("关节角度有效", not has_nan, "含 NaN" if has_nan else ""))

    if not has_nan:
        config = robot.arm.config
        in_range = bool(np.all(qpos >= config.qpos_min) and np.all(qpos <= config.qpos_max))
        checks.append(("关节在限位内", in_range, f"qpos={np.round(np.rad2deg(qpos), 1)}deg" if not in_range else ""))

    # ── Hand connection (degraded) ──
    hand_connected = robot.hand.is_connected()
    checks.append(("hand 连接", hand_connected or None, "" if hand_connected else "降级运行 (arm only)"))

    if hand_connected:
        hand_has_error = robot.hand.is_error()
        checks.append(("hand 通信正常", None if hand_has_error else True, "board error" if hand_has_error else ""))

    # ── Overall pass ──
    passed = all(ok is not False for _, ok, _ in checks)
    return PreFlightReport(passed=passed, checks=checks)


def print_preflight(report: PreFlightReport) -> None:
    """Print formatted pre-flight checklist results."""
    print("\nPre-Flight 检查:")
    for name, ok, detail in report.checks:
        if ok is True:
            status = "OK"
        elif ok is False:
            status = "FAIL"
        else:
            status = "WARN"
        detail_str = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
    has_warnings = any(ok is None for _, ok, _ in report.checks)
    if report.passed:
        result = "通过 (有告警)" if has_warnings else "全部通过"
    else:
        result = "检查失败，中止操作"
    print(f"\n结果: {result}\n")

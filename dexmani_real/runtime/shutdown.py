"""撤销运动权限并按既有顺序关闭本地运行资源。"""

import logging

from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)


def shutdown_local_runtime(
    runtime_state, robot=None, *, sensors=None, keyboard=None, timeout_s=5.0
):
    """Revoke, close devices and inputs, then stop sensors within their join budget."""
    with runtime_state.motion_lock:
        runtime_state.revoke_motion(
            (
                SafetyState.FAULT
                if runtime_state.safety_state == SafetyState.FAULT
                else SafetyState.DISARMED
            ),
            reason=RunEndReason.RUNTIME_SHUTDOWN,
        )
    for close in (
        robot.close if robot is not None else None,
        keyboard.stop if keyboard is not None else None,
    ):
        if close is None:
            continue
        try:
            close()
        except Exception:
            runtime_state.mark_error()
            logger.exception("local runtime cleanup failed")
    runtime_state.end_session()
    sensors_clean = True
    if sensors is not None:
        try:
            sensors_clean = sensors.shutdown(graceful_timeout_s=timeout_s)
        except Exception:
            runtime_state.mark_error()
            logger.exception("sensor shutdown failed; unresolved process handles retained")
            sensors_clean = False
    clean = (
        sensors_clean
        and not runtime_state.error_state
        and not runtime_state.estop_request
        and runtime_state.safety_state != SafetyState.FAULT
    )
    if not clean and not runtime_state.estop_request:
        runtime_state.mark_error()
    # DISARMED describes software authority only, never physical stopping.
    runtime_state.revoke_motion(SafetyState.DISARMED if clean else SafetyState.FAULT)
    return clean

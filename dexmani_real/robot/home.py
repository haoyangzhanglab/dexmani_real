"""Results for dedicated ARMED home operations, separate from normal streaming."""

import time
from dataclasses import dataclass
from queue import Empty

from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk, revoke_motion_if_run_id


@dataclass(frozen=True)
class HomeResult:
    ok: bool
    reason: str = ""


def wait_home_result(shared, results, run_id, timeout_s, abort_requested=None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if abort_requested is not None and abort_requested():
            break
        if not command_may_cross_sdk(
            shared, run_id=run_id, required_safety_state=SafetyState.ARMED
        ):
            return HomeResult(False, "home interrupted")
        try:
            epoch, result = results.get(timeout=0.01)
        except Empty:
            continue
        if epoch == run_id:
            if not command_may_cross_sdk(
                shared, run_id=run_id, required_safety_state=SafetyState.ARMED
            ):
                return HomeResult(False, "home interrupted")
            return result
    revoke_motion_if_run_id(shared, run_id)
    return HomeResult(False, "home aborted or timed out")

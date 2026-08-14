"""Offline smoke check: import the modules/functions under change without a device.

This repository has no conventional unit-test suite.  This check only proves
that the touched modules import cleanly (no syntax or import-time error) and
that the pure helpers we exercise elsewhere are present.  It never imports or
instantiates a hardware SDK: the xArm/XHand SDKs are imported lazily inside
their owning worker functions, not at module level.

Run from the repo root:
    conda run -n real_robot python checks/offline/check_imports.py
"""

from __future__ import annotations


def main() -> int:
    # robot / arm SDK-contract helpers
    from dexmani_real.robot.arm_loop import (  # noqa: F401
        ArmLoopConfig,
        _disconnect_arm,
        _enter_mode6_ready,
        _read_live_error_code,
        _read_live_status,
        _recover_c24_measured_hold,
        _require_sdk_ok,
        _wait_live_status,
        arm_loop,
    )

    # robot / hand
    from dexmani_real.robot.xhand import XHand  # noqa: F401
    from dexmani_real.robot.hand_process import hand_loop  # noqa: F401

    # config
    from dexmani_real.config.defaults import arm, hand, safety  # noqa: F401

    # IPC ring
    from dexmani_real.shm.ring_buffer import (  # noqa: F401
        CameraRingBuffer,
        SeqlockSlot,
        SharedMemoryRingBuffer,
    )

    # action protocol
    from dexmani_real.policy.safety import (  # noqa: F401
        publish_hand_home_and_wait_applied,
        publish_joint_targets,
        send_command,
    )

    print("OK: imports resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

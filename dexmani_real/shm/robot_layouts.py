"""Numpy dtype definitions for the arm/hand process-isolation SHM interface.

Authoritative source: docs/arm-hand-process-isolation-plan.md §4 — the field
names, order, and kinds below are verbatim from plan §4.1-4.5 (arm_state,
arm_target, arm_cmd/result, hand_state, hand_cmd) and §4.10 (policy_chunk)
and must not diverge. These records are placed in SharedMemoryRingBuffer /
SeqlockRingBuffer slots ([timestamp_ns, sequence, data] per slot — timestamps
live in the slot header, not in these records) and in the RPC layer
(arm_cmd / arm_cmd_result rings).

Layout principles (same as layouts.py):
  - Little-endian (<) for x86_64 compatibility
  - All arrays are flat (no objects, no variable-length strings)
  - Packed dtypes (no align=True) so the SHM layout matches the plan byte-for-byte
"""

from __future__ import annotations

import numpy as np

# ── Arm command codes (ARM_CMD_DTYPE["cmd"]) — plan §4.3 ──
ARM_CMD_EXEC_WAYPOINTS = 1  # Mode 1 set_servo_angle_j per waypoint (>2048 pts segmented by caller)
ARM_CMD_RESET_BLOCKING = 2  # Mode 0 set_servo_angle(wait=True) — home semantics unchanged
ARM_CMD_CLEAR_ERROR = 3
ARM_CMD_EMERGENCY_STOP = 4  # set_state(4)
ARM_CMD_REINIT_MODE6 = 5

# ── Hand macro codes (RPC) — plan §4.6 ──
HAND_MACRO_RESET = 1
HAND_MACRO_STOP = 2
HAND_MACRO_CLEAR_ERROR = 3
HAND_MACRO_SEND_TRAJECTORY = 4

# ── Producer IDs (arm_target / hand_cmd "producer_id") — plan §4.2 / D9 ──
PRODUCER_TELEOP = 1
PRODUCER_REPLAY = 2
PRODUCER_POLICY = 3

# ── Capacity limits ──
MAX_ARM_WAYPOINTS = 2048  # ARM_CMD_DTYPE waypoint rows (plan §4.3)
MAX_HAND_WAYPOINTS = 256  # SEND_TRAJECTORY waypoint rows (plan §4.6)

# ── arm_state (child → main / policy read-only, maxlen=3) — plan §4.1 ──
# Written every inner-loop tick after get_joint_states.
ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (7,)),
        ("qvel", "<f8", (7,)),
        ("tau", "<f8", (7,)),
        ("temps", "<f8", (7,)),  # qvel/tau/temps NaN until first valid readback
        ("error_state", "u1"),
        ("connected", "u1"),
        ("mode", "i4"),
        ("tracking_err", "<f8"),
        ("last_sent", "<f8", (7,)),  # inner-loop actual send (post delta-clip) → §4.9 sent stream
    ]
)

# ── arm_target (main → child, maxlen=2) — plan §4.2 ──
ARM_TARGET_DTYPE = np.dtype(
    [
        ("target", "<f8", (7,)),
        ("is_hold", "u1"),  # 1 = hold sentinel (set_target(None))
        ("producer_id", "u4"),  # 1=teleop 2=replay 3=policy; nonzero mismatch → reject + warn (D9)
    ]
)

# ── arm_cmd / arm_cmd_result (RPC, maxlen=2 each) — plan §4.3 ──
ARM_CMD_DTYPE = np.dtype(
    [
        ("cmd", "u4"),  # ARM_CMD_* codes above
        ("n_waypoints", "u4"),
        ("waypoints", "<f8", (MAX_ARM_WAYPOINTS, 7)),  # ~114KB/slot; dense home path typically <360 pts
        ("dt", "<f8"),
        ("target", "<f8", (7,)),
        ("speed", "<f8"),
        ("acc", "<f8"),
    ]
)

ARM_CMD_RESULT_DTYPE = np.dtype(
    [
        ("cmd_seq", "u8"),  # echoes the arm_cmd ring sequence this result answers
        ("ok", "u1"),
        ("arm_err", "i4"),
        ("sdk_ret", "i4"),
        ("final_qpos", "<f8", (7,)),
    ]
)

# ── hand_state (child → main / policy read-only, maxlen=3) — plan §4.4 ──
HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (12,)),
        ("current", "<f8", (12,)),  # motor current (mA); named 'torque' in SDK but carries mA
        ("temperature", "<f8", (12,)),  # motor temperature (°C); mirrors ARM_STATE_DTYPE temps
        ("last_qpos_cmd", "<f8", (12,)),  # value actually sent to hardware (post joint-limit net)
        ("last_cmd_seq", "u8"),  # echoes the hand_cmd ring sequence processed
        ("tactile_sum", "<f8", (5, 3)),
        ("tactile_force", "<f8", (5, 120, 3)),  # 14.4KB/frame, full recording bandwidth (D3)
        ("tactile_contact", "u1", (5,)),  # per-finger contact boolean (from detect_contact)
        ("tipboard_err", "<i4", (12,)),  # tip board error registers per joint
        ("connected_flag", "u1"),  # raw SDK flag (not composited with error_state)
        ("error_state", "u1"),
        ("consecutive_errs", "u4"),
        ("last_error_code", "i8"),
        ("limit_clipped", "u1"),
    ]
)

# ── hand_cmd (main → child, maxlen=8) — plan §4.5 (F1: clip applied main-side) ──
HAND_CMD_DTYPE = np.dtype(
    [
        ("qpos_cmd", "<f8", (12,)),  # target qpos (safety clipping by child)
        ("producer_id", "u4"),
    ]
)



def new_frame(dtype: np.dtype) -> np.ndarray:
    """Allocate a zero-initialized 1-element record of ``dtype`` for ring writes."""
    return np.zeros(1, dtype=dtype)

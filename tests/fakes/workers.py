"""Deterministic ring writers — fake sensor/robot workers for loop tests.

Each writer builds a single-frame structured array matching the ring's dtype,
fills the fields a loop consumer validates, and publishes through the ring's
seqlock ``write``.  Freshness fields default to ``time.monotonic_ns()``; pass an
explicit ``source_monotonic_ns`` to simulate stale feedback.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.shm.shared_storage import new_frame
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_CONTROL_DTYPE,
    ARM_STATE_DTYPE,
    HAND_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
    VR_FRAME_DTYPE,
)

_ARM_DOF = 7
_HAND_DOF = 12


def _seq(shared: Any, ring_name: str) -> int:
    ring = getattr(shared, ring_name)
    latest = getattr(ring, "latest_sequence", 0)
    return int(latest) + 1


def write_vr_frame(
    shared: Any,
    *,
    wrist_pos: Any,
    wrist_quat_wxyz: Any = (1.0, 0.0, 0.0, 0.0),
    landmarks: Any | None = None,
    local_recv_ns: int | None = None,
) -> int:
    """Write one VR frame; landmarks default to a neutral flat-hand layout."""
    frame = new_frame(VR_FRAME_DTYPE)
    now = time.monotonic_ns()
    if landmarks is None:
        # 21 hand landmarks laid out flat in front of the wrist; consumers only
        # require finiteness and correct shape (retargeters validate the rest).
        landmarks = np.column_stack(
            (
                np.linspace(-0.05, 0.05, 21),
                np.linspace(0.05, 0.20, 21),
                np.zeros(21),
            )
        )
    frame["wrist_pos"][0] = np.asarray(wrist_pos, dtype=np.float64)
    frame["wrist_quat_wxyz"][0] = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    frame["landmarks"][0] = np.asarray(landmarks, dtype=np.float64)
    frame["head_pos"][0] = np.zeros(3, dtype=np.float64)
    frame["head_quat_wxyz"][0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    recv_ns = local_recv_ns if local_recv_ns is not None else now
    seq = _seq(shared, "vr_ring")
    frame["recv_ts_ns"][0] = recv_ns
    frame["source_ts_ns"][0] = recv_ns
    frame["sequence_id"][0] = seq
    frame["source_frame_seq"][0] = seq
    frame["local_recv_ns"][0] = recv_ns
    frame["side"][0] = 0
    frame["head_sequence_id"][0] = seq
    frame["head_recv_ts_ns"][0] = recv_ns
    shared.vr_ring.write(frame)
    return seq


def write_arm_state(
    shared: Any,
    *,
    qpos: Any,
    qvel: Any | None = None,
    error_code: int = 0,
    connected: bool = True,
    state_valid: bool = True,
    last_cmd_seq: int = 0,
    source_monotonic_ns: int | None = None,
) -> None:
    """Write one arm state frame to the arm state ring."""
    frame = new_frame(ARM_STATE_DTYPE)
    now = time.monotonic_ns()
    src = now if source_monotonic_ns is None else int(source_monotonic_ns)
    frame["qpos"][0] = np.asarray(qpos, dtype=np.float64)
    frame["qvel"][0] = (
        np.zeros(_ARM_DOF) if qvel is None else np.asarray(qvel, dtype=np.float64)
    )
    frame["tau"][0] = np.zeros(_ARM_DOF, dtype=np.float64)
    frame["eef_pos"][0] = np.zeros(3, dtype=np.float64)
    frame["eef_rot6d"][0] = np.zeros(6, dtype=np.float64)
    frame["error_code"][0] = int(error_code)
    frame["connected"][0] = int(connected)
    frame["mode"][0] = 6
    frame["tracking_err"][0] = 0.0
    frame["last_cmd_seq"][0] = int(last_cmd_seq)
    frame["last_cmd_created_s"][0] = 0.0
    frame["last_cmd_received_s"][0] = 0.0
    frame["last_cmd_applied_s"][0] = 0.0
    frame["last_cmd_queue_latency_s"][0] = 0.0
    frame["last_cmd_apply_latency_s"][0] = 0.0
    frame["last_cmd_sdk_duration_s"][0] = 0.0
    frame["last_cmd_is_hold"][0] = 0
    frame["source_monotonic_ns"][0] = src
    frame["publish_monotonic_ns"][0] = now
    frame["state_valid"][0] = int(state_valid)
    frame["timestamp"][0] = src / 1e9
    shared.arm_state_ring.write(frame)


def write_hand_state(
    shared: Any,
    *,
    qpos: Any,
    connected: bool = True,
    error_state: bool = False,
    state_valid: bool = True,
    send_healthy: bool = True,
    read_healthy: bool = True,
    last_cmd_seq: int = 0,
    source_monotonic_ns: int | None = None,
) -> None:
    """Write one hand state frame to the hand state ring."""
    frame = new_frame(HAND_STATE_DTYPE)
    now = time.monotonic_ns()
    src = now if source_monotonic_ns is None else int(source_monotonic_ns)
    qpos_arr = np.asarray(qpos, dtype=np.float64)
    frame["qpos"][0] = qpos_arr
    frame["current"][0] = np.zeros(_HAND_DOF, dtype=np.float64)
    frame["tactile_sum"][0] = np.zeros((5, 3), dtype=np.float64)
    frame["tactile_contact"][0] = np.zeros(5, dtype=bool)
    frame["error_state"][0] = int(error_state)
    frame["connected"][0] = int(connected)
    frame["qpos_stale"][0] = 0
    frame["last_cmd_seq"][0] = int(last_cmd_seq)
    frame["last_cmd_qpos"][0] = qpos_arr
    frame["commboard_err"][0] = np.zeros(_HAND_DOF, dtype=np.int32)
    frame["jointboard_err"][0] = np.zeros(_HAND_DOF, dtype=np.int32)
    frame["tipboard_err"][0] = np.zeros(_HAND_DOF, dtype=np.int32)
    frame["source_monotonic_ns"][0] = src
    frame["publish_monotonic_ns"][0] = now
    frame["state_valid"][0] = int(state_valid)
    frame["send_healthy"][0] = int(send_healthy)
    frame["read_healthy"][0] = int(read_healthy)
    frame["timestamp"][0] = src / 1e9
    shared.hand_state_ring.write(frame)


def make_arm_command(
    shared: Any,
    qpos: Any,
    *,
    action_id: int,
    run_generation: int | None = None,
    is_hold: bool = False,
    valid_until_s: float = 10.0,
) -> np.ndarray:
    """Build a well-formed ARM_COMMAND_DTYPE record for the ordered queue."""
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    now = time.monotonic_ns()
    frame["run_generation"][0] = (
        int(shared.run_generation.value) if run_generation is None else int(run_generation)
    )
    frame["observation_id"][0] = int(action_id)
    frame["action_id"][0] = int(action_id)
    frame["created_monotonic_ns"][0] = now
    frame["target_monotonic_ns"][0] = now
    frame["valid_until_monotonic_ns"][0] = now + int(valid_until_s * 1e9)
    frame["is_hold"][0] = int(is_hold)
    frame["qpos_cmd"][0] = np.asarray(qpos, dtype=np.float64)
    return frame


def make_arm_control_request(
    shared: Any,
    kind: Any,
    *,
    action_id: int,
    run_generation: int | None = None,
    valid_until_s: float = 10.0,
) -> np.ndarray:
    """Build and publish a well-formed ARM_CONTROL_DTYPE record (latest-wins)."""
    frame = np.zeros(1, dtype=ARM_CONTROL_DTYPE)
    now = time.monotonic_ns()
    frame["kind"][0] = int(kind)
    frame["run_generation"][0] = (
        int(shared.run_generation.value) if run_generation is None else int(run_generation)
    )
    frame["action_id"][0] = int(action_id)
    frame["created_monotonic_ns"][0] = now
    frame["valid_until_monotonic_ns"][0] = now + int(valid_until_s * 1e9)
    shared.arm_control_ring.write(frame)
    return frame


def drain_arm_action_q(shared: Any, timeout_s: float = 0.0) -> Any | None:
    """Non-blocking pop of one item from the ordered arm action queue."""
    from queue import Empty

    try:
        return shared.arm_action_q.get(timeout=timeout_s)
    except Empty:
        return None
